# -*- coding: utf-8 -*-
"""
Subprotocol Extractor (Budgeted + Sharded + Merge)
- 证据只读（带 cites）
- 预算化拼包：限域/配额/条款级压缩/去重
- 自动分片：严格控 tokens
- 三段式 Prompt：system + developer(COT+Schema) + user
- 多分片 Map-Reduce 合并成唯一 JSON
"""
###2025-09-08 by maju
#### 调用模型：gpt-4o   A1

import os, re, json, math, logging, sys, hashlib, collections, uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple
from anthropic import Anthropic
import pytz

# tokenizer
try:
    import tiktoken
    _ENC = None
    def get_enc(model="gpt-4o"):
        global _ENC
        if _ENC is None:
            try:
                _ENC = tiktoken.encoding_for_model(model)
            except Exception:
                _ENC = tiktoken.get_encoding("cl100k_base")
        return _ENC
    def tk_len(s:str, model="gpt-4o"): return len(get_enc(model).encode(s))
except Exception:
    def tk_len(s:str, model="gpt-4o"): return max(1, len(s)//4)

from openai import OpenAI

# ----------------- 日志 -----------------
def setup_logging(log_file="subprotocol_prompt.log"):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear()
    fh=logging.FileHandler(log_file,mode="a",encoding="utf-8")
    ch=logging.StreamHandler(sys.stdout)
    fmt=logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logging.info(f"日志初始化：{log_file}")

def now(tz="Asia/Shanghai"):
    return datetime.now(pytz.timezone(tz))

# ----------------- I/O -----------------
def load_jsonl(path:str)->List[Dict[str,Any]]:
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)

# ----------------- 预处理与工具 -----------------
RFC2119 = re.compile(r"\b(MUST(?:\s+NOT)?|SHOULD(?:\s+NOT)?|MAY|REQUIRED|RECOMMENDED|NOT\s+RECOMMENDED|OPTIONAL|SHALL(?:\s+NOT)?)\b")
EVENT = re.compile(r"\b(upon|when|whenever|if|then|receive[sd]?|send[sd]?|ack|fin|timeout|expires?)\b", re.I)
ALLCAPS = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b") # 全大写的标识符
PURPOSE = re.compile(r"\b(provides|purpose|used to|controls|responsible|intended to|aims to)\b", re.I) # 目的句

STREAM_FRAMES = {"STREAM","RESET_STREAM","STOP_SENDING","STREAM_DATA_BLOCKED",
                 "DATA_BLOCKED","MAX_DATA","MAX_STREAM_DATA","MAX_STREAMS","STREAMS_BLOCKED"}

def simhash(text:str, ngram=3)->int:
    # SimHash 文本相似度，用于去重
    t=re.sub(r"\s+"," ",text.lower()).strip()
    grams=[t[i:i+ngram] for i in range(max(1,len(t)-ngram+1))]
    v=[0]*64
    for g in grams:
        h=int(hashlib.blake2b(g.encode("utf-8"),digest_size=8).hexdigest(),16)
        for b in range(64): v[b]+= 1 if (h>>b)&1 else -1
    out=0
    for b in range(64):
        if v[b]>0: out|=(1<<b)
    return out
#SimHash/Jaccard 去重，剔除高相似句/块
def hamdist(a:int,b:int)->int:
    x=a^b; c=0
    while x: x&=x-1; c+=1
    return c
#对长句只保留“模态词±窗口”与“帧/状态/事件”周边片段（不改变原文词序，仍可引用）
def compress_clause(text:str, frames:set, keep_chars:int=180)->str:
    spans=[]
    for m in RFC2119.finditer(text): spans.append((m.start(),m.end()))
    for m in EVENT.finditer(text): spans.append((m.start(),m.end()))
    for m in ALLCAPS.finditer(text):
        if m.group(1) in frames: spans.append((m.start(),m.end()))
    if not spans:
        return text[:keep_chars]+"…" if len(text)>keep_chars else text
    wins=[]
    for s,e in spans: wins.append((max(0,s-40), min(len(text),e+40)))
    wins.sort()
    merged=[]; cs,ce=wins[0]
    for s,e in wins[1:]:
        if s<=ce+16: ce=max(ce,e)
        else: merged.append((cs,ce)); cs,ce=s,e
    merged.append((cs,ce))
    parts=[text[s:e] for s,e in merged]
    out=" … ".join(parts)
    return out[:keep_chars]+"…" if len(out)>keep_chars else out

# ----------------- 预算化拼包（子协议阶段专用） -----------------
##   ContextPacket 生成脚本（读索引→筛选→拼包→输出 JSON）
def select_sections(sections:List[Dict[str,Any]])->List[Dict[str,Any]]:
    return [s for s in sections if any(s.get("section","").startswith(p) for p in ("2","3","4","19"))]

def build_toc(sections)->List[Dict[str,Any]]:
    return [{"section":s["section"],"title":s.get("title",""),"anchor":s.get("anchor",""),"order":s.get("order",0)} for s in sections]

def build_frame_index(sentences, sections):
    out=[]
    for sec in sections:
        if not sec["section"].startswith("19"): continue
        frames=set()
        for s in sentences:
            if s["section"].startswith(sec["section"]):
                for f in (s.get("entities",{}).get("frames") or []):
                    if f in STREAM_FRAMES: frames.add(f)
        if frames:
            out.append({"section":sec["section"],"title":sec["title"],"anchor":sec.get("anchor",""),"frames":sorted(frames)})
    return out

def build_budgeted_packet(sections_path:str, sentences_path:str,
                          doc_budget_tokens:int=4200,
                          min_sec_quota:int=120, max_sec_quota:int=480,
                          keep_per_sec:int=8, model="gpt-4o")->Dict[str,Any]:
    sections=load_jsonl(sections_path)
    sentences=load_jsonl(sentences_path)
    secs=select_sections(sections) # 只要 §2/§3/§4/§19.*
    toc=build_toc(secs) # 目录
    frame_idx=build_frame_index(sentences, secs) # 流相关帧索引
    # 标段首
    para_first=set()
    seen_para={}
    for s in sentences:
        pid=s.get("para_id"); sid=s.get("sent_id")
        if pid and pid not in seen_para:
            seen_para[pid]=sid; para_first.add(sid)

    # 每节收集候选
    buckets=collections.defaultdict(list) 
    for s in sentences:
        sec=s.get("section","")
        if not any(sec.startswith(x["section"]) for x in secs): continue
        txt=s.get("text",""); norm=s.get("normative",False); facet=s.get("facet","")
        score=0.0
        is_intro=(s.get("sent_id") in para_first or PURPOSE.search(txt) or facet.lower() in {"intro","purpose","streams","flow_control","state_machine","frames"})
        if is_intro: score+=3.0
        if norm and RFC2119.search(txt): score+=2.0
        ents=s.get("entities",{})
        if ents.get("frames"): score+=1.5
        if ents.get("states"): score+=1.0
        score/=max(10,len(txt))**0.5
        buckets[sec.split(".")[0]].append((score,s))

    # 节配额：对每个相关小节分配最小/最大 token 配额（如 min 80 / max 400），避免单章“吃满”
    uniq_secs=sorted({x["section"] for x in secs})
    per=doc_budget_tokens//max(1,len(uniq_secs))
    per=min(max(per,min_sec_quota),max_sec_quota)

    frames_set=set(STREAM_FRAMES)
    evidence=[] # 证据包
    used=0
    seen_hash=[]
    for sec_id in uniq_secs:
        # 候选排序
        items=[]
        for (sc,s) in buckets.get(sec_id,[]):
            items.append((sc,s))
        items.sort(key=lambda x:-x[0])
        take=0; acc=0; sec_block=[]
        for sc,s in items[:80]:
            txt=compress_clause(s["text"], frames_set, keep_chars=180)
            h=simhash(txt)
            if any(hamdist(h,h0)<=8 for h0 in seen_hash): continue
            toks=tk_len(txt, model)
            if acc+toks>per and take>0: break
            sec_block.append({
                "type":"purpose" if (s.get("sent_id") in para_first or PURPOSE.search(s["text"])) else "normative",
                "text":txt,
                "source":{"section":s.get("section"),"anchor":s.get("anchor",""),
                          "para":s.get("para_id"),"sent_id":s.get("sent_id")}
            })
            seen_hash.append(h); acc+=toks; take+=1
        if sec_block:
            title=next((x["title"] for x in secs if x["section"]==sec_id),"")
            anch =next((x.get("anchor","") for x in secs if x["section"]==sec_id),"")
            evidence.append({"section":sec_id,"title":title,"anchor":anch,"evidence":sec_block})
            used+=acc

    packet={"toc":toc,"evidence":evidence,"frame_evidence":frame_idx,
            "scope_hint":"RFC9000 §2/§3/§4/§19.*（导言/目的/规范句；流相关帧索引）"}  #scope_hint: 候选小节号列表
    return packet

# ----------------- 分片（严格控 tokens） -----------------
def messages_tokens(messages:List[Dict[str,str]], model="gpt-4o")->int:
    return sum(4 + tk_len(m.get("content",""), model) for m in messages)

def shard_packet(packet:Dict[str,Any], base_tokens:int, hard_cap:int, model="gpt-4o")->List[Dict[str,Any]]:
    """按 section 切成若干片，确保 base_tokens + packet_json_tokens <= hard_cap"""
    enc_budget = hard_cap - base_tokens - 256  # 给 JSON 语法/边距留余量
    # 先估一估整个包
    pj=json.dumps(packet, ensure_ascii=False, separators=(",",":"))
    if tk_len(pj, model) <= enc_budget:
        return [packet]

    # 分片：逐节累加   ##分片细节：是否与上一片有交集？
    shards=[]; 
    cur={
         "toc":[], 
         "evidence":[], 
         "frame_evidence":[], 
         "scope_hint":packet["scope_hint"]}
    cur_secs=set()
    toc_map={t["section"]:t for t in packet["toc"]}
    frame_by_sec=collections.defaultdict(list)
    for f in packet.get("frame_evidence",[]):
        frame_by_sec[f["section"]].append(f)

    def pkt_len(p):
        j=json.dumps(p, ensure_ascii=False, separators=(",",":"))
        return tk_len(j, model)

    for blk in packet["evidence"]:
        tentative={
            "toc":[toc_map[s] for s in cur_secs|{blk["section"]} if s in toc_map],
            "evidence":cur["evidence"]+[blk],
            "frame_evidence":[x for s in (cur_secs|{blk["section"]}) for x in frame_by_sec.get(s,[])],
            "scope_hint":packet["scope_hint"]
        }
        if pkt_len(tentative) <= enc_budget:
            cur=tentative; cur_secs.add(blk["section"])
        else:
            # 关闭当前片
            if cur["evidence"]:
                shards.append(cur)
            # 开新片
            cur={"toc":[toc_map.get(blk["section"],{})],
                 "evidence":[blk],
                 "frame_evidence":frame_by_sec.get(blk["section"],[]),
                 "scope_hint":packet["scope_hint"]}
            cur_secs={blk["section"]}
    if cur["evidence"]: 
        shards.append(cur)
    return shards

# ----------------- 三段式 Prompt（子协议阶段） -----------------
SYSTEM_PROMPT = """你是“协议规范编译器”。任务：仅依据提供的 RFC 证据（含章节号与锚点），
把 QUIC (RFC 9000) 的主要子协议划分出来，并给出用途、关键帧、依赖与边界。
铁规：
1) 仅使用输入 evidence；禁止外部知识或猜测。
2) 每条结论必须给 cites（section#anchor/para_id/sent_id）。
3) 允许内部推理，但最终只输出 JSON；不得输出推理过程。
4) 术语统一：client≈initiator，server≈responder，endpoint≈peer；帧名保持全大写。
5) 模态优先级：MUST/SHALL > SHOULD/RECOMMENDED > MAY/OPTIONAL；冲突时偏向强模态。
6) 证据不足时，输出 open_questions[] 指明缺口与应补小节/帧。
7) 严格符合 JSON Schema，勿添加未定义字段。"""

DEVELOPER_PROMPT = """【工作流】
A. 候选模块：从 toc + evidence 导言/目的/规范句中归纳主要子协议（Streams、Flow Control、Connection Management/Migration、Connection-level 传输、Connection Closure 等，前提是有证据）。
B. purpose：用导言/目的句凝练（≤200字符），附 1–3 条 cites。
C. 边界与依赖：boundaries.includes（小节号集合）、excludes（明确不含主题，如 RFC9001/9002）；depends_on（如 Streams 依赖 Flow Control）。
D. key_frames：把与该模块相关的帧名（仅名称）列出；证据可来自 §19.* 的帧定义或正文提及。
E. 覆盖检查：若 frame_evidence 里有帧未被任何模块收录，加入 open_questions。
【输出 JSON Schema】
{
  "type":"object",
  "required":["subprotocols","open_questions"],
  "properties":{
    "subprotocols":{"type":"array","items":{
      "type":"object",
      "required":["name","purpose","key_frames","depends_on","boundaries","cites"],
      "properties":{
        "name":{"type":"string"},
        "purpose":{"type":"string","maxLength":200},
        "key_frames":{"type":"array","items":{"type":"string"}},
        "depends_on":{"type":"array","items":{"type":"string"}},
        "boundaries":{"type":"object","properties":{
          "includes":{"type":"array","items":{"type":"string"}},
          "excludes":{"type":"array","items":{"type":"string"}}
        }},
        "cites":{"type":"array","items":{"type":"string"}}
      }
    }},
    "open_questions":{"type":"array","items":{
      "type":"object",
      "required":["ask"],
      "properties":{
        "ask":{"type":"string"},
        "related_frames":{"type":"array","items":{"type":"string"}},
        "wanted_sections":{"type":"array","items":{"type":"string"}}
      }
    }}
  }
}"""

def build_user_prompt(context_packet:Dict[str,Any])->str:
    j=json.dumps(context_packet, ensure_ascii=False, indent=2)
    return (
f"任务：在 RFC 证据范围内，抽取“主要子协议/主要部分”，用于后续选择建模范围。\n"
f"只输出符合上方 JSON Schema 的 JSON。每个字段都要给 cites（使用 evidence.source 的 section/anchor/para/sent_id）。\n\n"
f"=== ContextPacket (subprotocols) ===\n{j}\n" )


# ----------------- 合并多分片草案 -----------------
def merge_drafts(drafts:List[str])->Dict[str,Any]:
    out={"subprotocols":[],"open_questions":[]}
    by_name={}
    for dj in drafts:
        try: d=json.loads(dj)
        except Exception: continue
        for sp in d.get("subprotocols",[]):
            name=(sp.get("name") or "").strip()
            if not name: continue
            k=name.lower()
            cur=by_name.get(k, {"name":name,"purpose":"","key_frames":[],"depends_on":[],"boundaries":{"includes":[],"excludes":[]},"cites":[]})
            if len(sp.get("purpose","")) > len(cur["purpose"]): 
                cur["purpose"]=sp.get("purpose","")
            cur["key_frames"]=sorted(list(set(cur["key_frames"] + (sp.get("key_frames") or []))))
            cur["depends_on"]=sorted(list(set(cur["depends_on"] + (sp.get("depends_on") or []))))
            b=sp.get("boundaries",{}) or {}
            cur["boundaries"]["includes"]=sorted(list(set(cur["boundaries"]["includes"] + (b.get("includes") or []))))
            cur["boundaries"]["excludes"]=sorted(list(set(cur["boundaries"]["excludes"] + (b.get("excludes") or []))))
            cur["cites"]=sorted(list(set(cur["cites"] + (sp.get("cites") or []))))
            by_name[k]=cur
        out["open_questions"] += d.get("open_questions",[])
    # open_questions 未解决的问题
    seen=set(); oq=[]
    for q in out["open_questions"]:
        k=json.dumps(q,ensure_ascii=False,sort_keys=True)
        if k in seen: continue
        seen.add(k); oq.append(q)
    out["open_questions"]=oq
    out["subprotocols"]=list(by_name.values())
    return out

# ----------------- 主流程 -----------------
def main():
    setup_logging(); logging.info(f"启动时间：{now().strftime('%F %T %Z')}")

    # === 路径与模型参数 ===
    sections_path = os.getenv("SECTIONS_JSONL", "d:/ProtocolAutoModel-main/data/sections.jsonl")
    sentences_path = os.getenv("SENTENCES_JSONL", "d:/ProtocolAutoModel-main/data/sentences.jsonl")
    out_dir = os.path.join(os.getcwd(), "output", "agents"); os.makedirs(out_dir, exist_ok=True)

    MODEL = os.getenv("MODEL_NAME","gpt-4o")
    MAX_INPUT = int(os.getenv("MAX_INPUT_TOKENS","8192"))   # 你的后端窗口；8k/16k 请自行改
    RESERVE   = int(os.getenv("RESERVE_FOR_OUTPUT","1024")) 
    HARD_CAP  = MAX_INPUT - RESERVE
    logging.info(f"MODEL={MODEL}, MAX_INPUT={MAX_INPUT}, HARD_CAP={HARD_CAP}, MAX_OUTPUT_TOKENS={RESERVE}")


    # === 1) 预算化拼包（单包） ===
    packet = build_budgeted_packet(
        sections_path, sentences_path,
        doc_budget_tokens=int(os.getenv("DOC_BUDGET_TOKENS","2400")),
        min_sec_quota=int(os.getenv("MIN_SEC_QUOTA","80")),
        max_sec_quota=int(os.getenv("MAX_SEC_QUOTA","320")),
        keep_per_sec=int(os.getenv("KEEP_PER_SECTION","6")),
        model=MODEL
    )
    # 估算模板自身 token（system+developer）
    base_msgs=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"developer","content":DEVELOPER_PROMPT},
        {"role":"user","content":"PLACEHOLDER"}
    ]
    base_tokens = messages_tokens(base_msgs, MODEL)
    logging.info(f"模板基线 tokens ≈ {base_tokens}")

    # === 2) 分片：严格控 tokens ===
    shards = shard_packet(packet, base_tokens, HARD_CAP, MODEL)
    logging.info(f"分片数：{len(shards)}")

    # === 3) 逐片调用 ===   ##逐片调用，是否将上一片的调用结果与下一篇一起输入？这样的效果是否会更好？
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY",""),
        base_url=os.getenv("OPENAI_BASE_URL","https://api.gpt.ge/v1/")
    )

    # client = Anthropic(api_key=os.getenv("OPENAI_API_KEY","sk-po4PH2gUa11tTr4733118e1f07944a64Ac1f2bC9E8158602"),
    #                    base_url=os.getenv("ANTHROPIC_BASE_URL","https://api.anthropic.com/"))
    drafts=[]
    for i,pkt in enumerate(shards, start=1):
        user_prompt = build_user_prompt(pkt)
        # print(user_prompt)
        messages=[
            # {"role":"system","content":SYSTEM_PROMPT},
            # {"role":"developer","content":DEVELOPER_PROMPT},
            {"role":"system","content":SYSTEM_PROMPT + "\n\n--- Developer Guidelines ---\n" + DEVELOPER_PROMPT},
            {"role":"user","content":user_prompt},
        ]
        tot = messages_tokens(messages, MODEL)
        # 仍超则裁剪 evidence 数组直到满足
        if tot > HARD_CAP:
            logging.warning(f"[Shard {i}] 超预算，启动二次裁剪 evidence…")
            safe = dict(pkt)
            ev = list(safe.get("evidence",[]))
            while ev and messages_tokens([
                # {"role":"system","content":SYSTEM_PROMPT},
                #  {"role":"developer","content":DEVELOPER_PROMPT},
                {"role":"system","content":SYSTEM_PROMPT + "\n\n--- Developer Guidelines ---\n" + DEVELOPER_PROMPT},
                 {"role":"user","content":build_user_prompt({**safe,"evidence":ev})}], MODEL
            ) > HARD_CAP:
                ev = ev[:-1]  # 每次去掉一个 section block
            safe["evidence"]=ev
            user_prompt = build_user_prompt(safe)
            # print(f"[Shard {i}] 裁剪后 evidence 数量：{len(ev)}")
            # print(user_prompt)
            messages=[
                {"role":"system","content":SYSTEM_PROMPT + "\n\n--- Developer Guidelines ---\n" + DEVELOPER_PROMPT},
                # {"role":"system","content":SYSTEM_PROMPT},
                # {"role":"developer","content":DEVELOPER_PROMPT},
                {"role":"user","content":user_prompt}
            ]
        
        logging.info(f"[Shard {i}] 发送 tokens≈{messages_tokens(messages, MODEL)}")
        # print(messages[2])
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1024")),
            temperature=0
        )
        raw = rsp.choices[0].message.content or ""
        m = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        text = m.group(1).strip() if m else raw.strip()
        drafts.append(text)
        logging.info(f"[Shard {i}] 收到结果 {len(text)} chars")

    # === 4) 合并草案 ===
    merged = merge_drafts(drafts)
    out_path = os.path.join(out_dir, f"QUIC_subprotocols_{MODEL}.json")
    save_json(merged, out_path)
    print("===== 子协议提取（合并后）=====")
    print(json.dumps(merged, ensure_ascii=False, indent=2))
    logging.info(f"已保存：{out_path}")

if __name__=="__main__":
    main()
