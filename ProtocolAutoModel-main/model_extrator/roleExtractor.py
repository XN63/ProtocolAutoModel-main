# -*- coding: utf-8 -*-
"""
Roles Extractor (Budgeted + Sharded + Merge)
- 输入：sections.jsonl / sentences.jsonl + 已产出的 subprotocols.json（用于限定 includes）
- 输出：output/agents/QUIC_roles_<MODEL>.json
"""

import os, re, json, logging, sys, hashlib, collections
from typing import Any, Dict, List
from datetime import datetime
import pytz
from APIConfig import LLMConfig, create_llm

try:
    import tiktoken
    _ENC = None
    def get_enc(model="gpt-4o"):
        global _ENC
        if _ENC is None:
            try: _ENC = tiktoken.encoding_for_model(model)
            except Exception: _ENC = tiktoken.get_encoding("cl100k_base")
        return _ENC
    def tk_len(s:str, model="gpt-4o"): return len(get_enc(model).encode(s))
except Exception:
    def tk_len(s:str, model="gpt-4o"): return max(1, len(s)//4)

def setup_logging(log_file="role_prompt.log"):
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
    import pytz
    return datetime.now(pytz.timezone(tz))

def load_jsonl(path:str)->List[Dict[str,Any]]:
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)

RFC2119 = re.compile(r"\b(MUST(?:\s+NOT)?|SHOULD(?:\s+NOT)?|MAY|REQUIRED|RECOMMENDED|NOT\s+RECOMMENDED|OPTIONAL|SHALL(?:\s+NOT)?)\b")
EVENT = re.compile(r"\b(upon|when|whenever|if|then|receive[sd]?|send[sd]?|ack|timeout|expires?)\b", re.I)
PURPOSE = re.compile(r"\b(provides|purpose|used to|controls|responsible|intended to|aims to)\b", re.I)

ROLE_LEX = re.compile(r"\b(client|server|endpoint|peer|sender|receiver|initiator|responder|the sending part of the stream)|the receving part of the stream|The sender of a stream|\b", re.I)

def simhash(text:str, ngram=3)->int:
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

def hamdist(a:int,b:int)->int:
    x=a^b; c=0
    while x: x&=x-1; c+=1
    return c

def compress_clause(text:str, keep_chars:int=180)->str:
    # 角色：更看重规范 + 事件动词，压缩窗口 ±40
    spans=[]
    for m in RFC2119.finditer(text): spans.append((m.start(),m.end()))
    for m in EVENT.finditer(text): spans.append((m.start(),m.end()))
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

def messages_tokens(messages:List[Dict[str,str]], model="gpt-4o")->int:
    return sum(4 + tk_len(m.get("content",""), model) for m in messages)

# ========= 拼包（角色阶段） =========
def build_roles_packet(sections_path:str, sentences_path:str, subproto_json:str, llm_conf:LLMConfig, model:str)->Dict[str,Any]:
    sections=load_jsonl(sections_path)
    sentences=load_jsonl(sentences_path)
    subp = json.load(open(subproto_json,"r",encoding="utf-8"))
    # 找到目标子协议（ENV: SUBP_NAME），并收集 includes 小节列表
    subp_name = os.getenv("SUBP_NAME","Streams")
    target = None
    for sp in subp.get("subprotocols",[]):
        if sp.get("name","").lower()==subp_name.lower():
            target = sp; break
    if not target:
        # 退化：不限定 includes
        includes = sorted(set([s.get("section","").split(".")[0] for s in sections if s.get("section")]))
    else:
        includes = sorted(set([sec.split(".")[0] for sec in target.get("boundaries",{}).get("includes",[])])) or \
                   sorted(set([s.get("section","").split(".")[0] for s in sections if s.get("section")]))
    # 目录
    toc=[{"section":s["section"],"title":s.get("title",""),"anchor":s.get("anchor","")} for s in sections if s.get("section","").split(".")[0] in includes]

    # 句子候选：角色词 +（规范/事件）打分
    buckets=collections.defaultdict(list)
    seen_para={}
    para_first=set()
    for s in sentences:
        pid=s.get("para_id"); sid=s.get("sent_id")
        if pid and pid not in seen_para:
            seen_para[pid]=sid; para_first.add(sid)
    for s in sentences:
        sec=s.get("section",""); sec0=sec.split(".")[0]
        if sec0 not in includes: continue
        txt=s.get("text","")
        if not ROLE_LEX.search(txt): 
            continue
        score = 0.0
        if s.get("sent_id") in para_first or PURPOSE.search(txt): score += 1.0
        if RFC2119.search(txt): score += 2.0
        if EVENT.search(txt): score += 1.0
        score /= max(10,len(txt))**0.5
        buckets[sec0].append((score,s))

    # 配额（使用模型自带预算）
    per = llm_conf.MODEL_CONFIGS[model].doc_budget_tokens // max(1,len(includes))
    per = min(max(per, llm_conf.MODEL_CONFIGS[model].min_sec_quota),
              llm_conf.MODEL_CONFIGS[model].max_sec_quota)

    evidence=[]; seen_hash=[]; 
    for sec0 in includes:
        items = sorted(buckets.get(sec0,[]), key=lambda x:-x[0])[:120]
        acc=0; block=[]
        for sc,s in items:
            txt = compress_clause(s["text"])
            h=simhash(txt)
            if any(hamdist(h,h0)<=8 for h0 in seen_hash): 
                continue
            toks=tk_len(txt, model)
            if acc + toks > per and block: break
            block.append({
                "type":"role_capability",
                "text":txt,
                "source":{"section":s.get("section"),"anchor":s.get("anchor",""),
                          "para":s.get("para_id"),"sent_id":s.get("sent_id")}
            })
            seen_hash.append(h); acc+=toks

        if block:
            title = next((x["title"] for x in sections if x["section"]==sec0),"")
            anch  = next((x.get("anchor","") for x in sections if x["section"]==sec0),"")
            evidence.append({"section":sec0,"title":title,"anchor":anch,"evidence":block})

    packet={"subprotocol": subp_name,
            "toc": toc,
            "evidence": evidence,
            "scope_hint":"角色/能力（规范 + 事件动词优先）"}
    return packet

# ========= 三段式提示（角色） =========
SYSTEM_PROMPT = """你是“协议规范编译器”。
任务：仅依据提供的 RFC 证据（含章节号与锚点），
抽取目标子协议中的参与方（角色）及其能力/职责（capabilities），并输出严格 JSON。
铁规：
1) 只能使用输入 evidence；禁止外部知识或猜测。
2) 每条结论必须给 cites（section#anchor/para_id/sent_id）。
3) 允许内部推理，但最终只输出 JSON；不得输出思维过程。
4) 统一同义词：client≈initiator，server≈responder，endpoint≈peer；sender/receiver 为角色。
5) 模态优先级：MUST/SHALL > SHOULD/RECOMMENDED > MAY/OPTIONAL；冲突时偏向强模态。
"""

DEVELOPER_PROMPT = """【工作流】
A. 在子协议范围内，列出出现于 evidence 中的角色（client/server/endpoint/peer/sender/receiver/initiator/responder）。
B. 为每个角色抽 capabilities：用“动词 + 对象”短语表述（如 "MUST send STREAM"、"MAY update flow credit"），每条附 cites。
C. 避免臆造；未在证据中出现的不输出。
D.cites 必须是字符串数组，仅包含来自 cites_pool 的 eid。禁止使用 section/anchor 文本；禁止自造 eid
【输出 JSON Schema】
{
  "type":"object",
  "required":["subprotocol","roles"],
  "properties":{
    "subprotocol":{"type":"string"},
    "roles":{"type":"array","items":{
      "type":"object",
      "required":["name","capabilities","cites"],
      "properties":{
        "name":{"type":"string"},
        "capabilities":{"type":"array","items":{"type":"string"}},
        "cites":{"type":"array","items":{"type":"string"}}
      }
    }}
  }
}
仅输出 JSON。
"""

def build_user_prompt(context_packet:Dict[str,Any])->str:
    j=json.dumps(context_packet, ensure_ascii=False, indent=2)
    return ("任务：抽取目标子协议的参与方与其能力/职责（capabilities）。\n"
            "请严格按上方 JSON Schema 输出，并为每个字段给 cites。\n\n"
            f"=== ContextPacket (roles) ===\n{j}\n")

# ========= 合并（角色） =========
def merge_roles_drafts(drafts: List[str]) -> Dict[str,Any]:
    subp=""; roles={}
    for dj in drafts:
        try: d=json.loads(dj)
        except: continue
        subp = subp or d.get("subprotocol","")
        for r in d.get("roles",[]):
            name=(r.get("name") or "").strip()
            if not name: continue
            k=name.lower()
            cur=roles.get(k, {"name":name,"capabilities":[],"cites":[]})
            cur["capabilities"] = list(dict.fromkeys(cur["capabilities"] + (r.get("capabilities") or [])))
            cur["cites"] = list(dict.fromkeys(cur["cites"] + (r.get("cites") or [])))
            roles[k]=cur
    return {"subprotocol": subp, "roles": list(roles.values())}

# ========= 分片 & 调用 =========
def shard_packet(packet:Dict[str,Any], base_tokens:int, hard_cap:int, model="gpt-4o")->List[Dict[str,Any]]:
    enc_budget = hard_cap - base_tokens - 256
    pj=json.dumps(packet, ensure_ascii=False, separators=(",",":"))
    if tk_len(pj, model) <= enc_budget: return [packet]
    shards=[]; cur={"subprotocol":packet["subprotocol"],"toc":[],"evidence":[],"scope_hint":packet["scope_hint"]}
    toc_map={t["section"]:t for t in packet["toc"]}
    def pkt_len(p): return tk_len(json.dumps(p, ensure_ascii=False, separators=(",",":")), model)
    for blk in packet["evidence"]:
        sec=blk["section"]
        tentative={"subprotocol":packet["subprotocol"],"toc":cur["toc"]+[toc_map.get(sec,{"section":sec})],
                   "evidence":cur["evidence"]+[blk],"scope_hint":packet["scope_hint"]}
        if pkt_len(tentative)<=enc_budget:
            cur=tentative
        else:
            if cur["evidence"]: shards.append(cur)
            cur={"subprotocol":packet["subprotocol"],"toc":[toc_map.get(sec,{"section":sec})],
                 "evidence":[blk],"scope_hint":packet["scope_hint"]}
    if cur["evidence"]: shards.append(cur)
    return shards

# ========= 主流程 =========
def main():
    
    setup_logging(); 
    logging.info(f"启动时间：{now().strftime('%F %T %Z')}")
#/root/project/ProtocolMAS/protocolMas/data/rfc9000_corpus/sections.jsonl
    sections_path = os.getenv("SECTIONS_JSONL", "d:/ProtocolAutoModel-main/data/rfc9000_corpus/sections.jsonl")
    sentences_path = os.getenv("SENTENCES_JSONL", "d:/ProtocolAutoModel-main/data/rfc9000_corpus/sentences.jsonl")
    subp_json = os.getenv("SUBP_JSON", "./output/agents/subprotocol/QUIC_subprotocols_gpt-4o_developpromptchange.json")
    MODEL = os.getenv("MODEL_NAME","qwen2-5-72b-instruct") # gpt-4o / gpt-4o-mini / qwen2-5-72b-instruct

    out_dir = os.path.join(os.getcwd(),"output","agents","role"); os.makedirs(out_dir, exist_ok=True)

    llm_cfg = LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY","sk-po4PH2gUa11tTr4733118e1f07944a64Ac1f2bC9E8158602"),
        base_url=os.getenv("OPENAI_BASE_URL","https://api.gpt.ge/v1/"),
        model_name=MODEL
    )
    MAX_INPUT = llm_cfg.max_input_tokens
    HARD_CAP  = llm_cfg.hard_cap
    RESERVE = llm_cfg.max_output_tokens

    logging.info(f"MODEL={MODEL}, MAX_INPUT={MAX_INPUT}, HARD_CAP={HARD_CAP}, MAX_OUTPUT_TOKENS={RESERVE}")
    packet = build_roles_packet(
        sections_path, 
        sentences_path, 
        subp_json, 
        llm_cfg, 
        MODEL)

    base_msgs=[{"role":"system","content":SYSTEM_PROMPT},
               {"role":"developer","content":DEVELOPER_PROMPT},
               {"role":"user","content":"PLACEHOLDER"}]
    
    base_tokens = messages_tokens(base_msgs, MODEL)
    logging.info(f"模板基线 tokens ≈ {base_tokens}")

    shards = shard_packet(packet, base_tokens, HARD_CAP, MODEL)
    logging.info(f"分片数：{len(shards)}")

    llm = create_llm(MODEL, llm_cfg)
    drafts=[]
    for i,pkt in enumerate(shards,1):
        user_prompt = build_user_prompt(pkt)
        messages = llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT, user_prompt)
        # 裁剪确保不超
        tot = messages_tokens(messages, MODEL)
        if tot>HARD_CAP:
            ev=list(pkt["evidence"])
            while ev and messages_tokens(
                llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT,
                                    build_user_prompt({**pkt,"evidence":ev})), MODEL) > HARD_CAP:
                ev = ev[:-1]
            pkt = {**pkt,"evidence":ev}
            messages = llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT, build_user_prompt(pkt))
        
        logging.info(f"[Shard {i}] 发送 tokens≈{messages_tokens(messages, MODEL)}")

        rsp = llm.call(messages, max_tokens=llm_cfg.max_output_tokens)
        m=re.search(r'```json\s*(.*?)\s*```', rsp, re.S)
        text=(m.group(1) if m else rsp).strip()
        drafts.append(text)

        logging.info(f"[Shard {i}] 收到结果 {len(text)} chars")
        
    merged = merge_roles_drafts(drafts)
    out_path = os.path.join(out_dir, f"QUIC_roles_{MODEL}_0911.json")
    save_json(merged, out_path)
    print(json.dumps(merged, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
