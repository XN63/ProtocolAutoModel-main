# -*- coding: utf-8 -*-
"""
Transitions Extractor (Budgeted + Sharded + Merge)
- 模式：ENV TRANSITION_MODE = frame / mech
- 输入：sections.jsonl / sentences.jsonl + subprotocols.json（取 key_frames / includes）
- 输出：output/agents/QUIC_transitions_<MODE>_<MODEL>.json
"""
"""
20250909 问题记录：
-transtions的抽取逻辑不太对，基于帧和机制的抽取思路不清晰，机制的抽取不仅涉及帧的触发，还有相应的参数
-对于状态抽取结果的应用，存在两种方案需要进行评估：
  1. 直接使用抽取的状态转换结果，可能会因为抽取不完整或不准确而影响后续的分析和验证。导致转换的抽取依赖于状态的抽取
  2. 将抽取的状态转换结果作为辅助信息，结合原始文本进行综合分析，以提高准确性和完整性。
-需要进一步明确状态转换的定义和范围，确保抽取的内容符合预期。
-需要对抽取结果进行人工审核和修正，以提升整体质量。
-需要考虑不同模型在处理复杂协议文本时的表现差异，选择最适合的模型进行抽取。
-需要优化提示词设计，使其更明确地指导模型进行状态转换的抽取。
-需要评估抽取结果在实际应用中的效果，确保其能够满足协议分析和验证的需求。   
"""
from datetime import datetime
import os, re, json, logging, sys, hashlib, collections
from typing import Any, Dict, List
from APIConfig import LLMConfig, create_llm

try:
    import tiktoken
    _ENC=None
    def get_enc(model="gpt-4o"):
        global _ENC
        if _ENC is None:
            try:_ENC=tiktoken.encoding_for_model(model)
            except Exception:_ENC=tiktoken.get_encoding("cl100k_base")
        return _ENC
    def tk_len(s, model="gpt-4o"): return len(get_enc(model).encode(s))
except Exception:
    def tk_len(s, model="gpt-4o"): return max(1,len(s)//4)

def load_jsonl(p):
    with open(p,"r",encoding="utf-8") as f: 
        return [json.loads(l) for l in f if l.strip()]

def save_json(o,p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p,"w",encoding="utf-8") as f: json.dump(o,f,ensure_ascii=False,indent=2)

def now(tz="Asia/Shanghai"):
    import pytz
    return datetime.now(pytz.timezone(tz))


def setup_logging(log_file="transitions_prompt.log"):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear()
    fh=logging.FileHandler(log_file,mode="a",encoding="utf-8")
    ch=logging.StreamHandler(sys.stdout)
    fmt=logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logging.info(f"日志初始化：{log_file}")


RFC2119 = re.compile(r"\b(MUST(?:\s+NOT)?|SHOULD(?:\s+NOT)?|MAY|SHALL(?:\s+NOT)?)\b")
EVENT = re.compile(r"\b(upon|when|if|then|receive[sd]?|send[sd]?|ack|fin|blocked|timeout|expires?)\b", re.I)
ALLCAPS = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
STREAM_FRAMES = {"STREAM","RESET_STREAM","STREAM_DATA_BLOCKED",
                 "DATA_BLOCKED","MAX_DATA","MAX_STREAM_DATA","MAX_STREAMS","STREAMS_BLOCKED","STOP_SENDING"}
MECH_LEX = re.compile(r"\b(flow\s*control|credit|window|limit|blocked|congestion|cwnd|ssthresh|rtt|pto|ecn|loss|recovery|max_[a-z_]+|_blocked)\b", re.I)

def simhash(t, ngram=3):
    t=re.sub(r"\s+"," ",t.lower()).strip()
    grams=[t[i:i+ngram] for i in range(max(1,len(t)-ngram+1))]
    v=[0]*64
    import hashlib
    for g in grams:
        h=int(hashlib.blake2b(g.encode("utf-8"),digest_size=8).hexdigest(),16)
        for b in range(64): v[b]+=1 if (h>>b)&1 else -1
    x=0
    for b in range(64):
        if v[b]>0: x|=(1<<b)
    return x

def ham(a,b):
    x=a^b; c=0
    while x: x&=x-1; c+=1
    return c

def compress_clause(text, keep=200):
    spans=[]
    for m in RFC2119.finditer(text): spans.append((m.start(),m.end()))
    for m in EVENT.finditer(text): spans.append((m.start(),m.end()))
    for m in ALLCAPS.finditer(text):
        if m.group(1) in STREAM_FRAMES: spans.append((m.start(),m.end()))
    if not spans: return text[:keep]+"…" if len(text)>keep else text
    wins=[]
    for s,e in spans: wins.append((max(0,s-40),min(len(text),e+40)))
    wins.sort()
    merged=[]; cs,ce=wins[0]
    for s,e in wins[1:]:
        if s<=ce+16: ce=max(ce,e)
        else: merged.append((cs,ce)); cs,ce=s,e
    merged.append((cs,ce))
    parts=[text[s:e] for s,e in merged]
    out=" … ".join(parts)
    return out[:keep]+"…" if len(out)>keep else out

def messages_tokens(messages, model="gpt-4o"):
    return sum(4 + tk_len(m.get("content",""), model) for m in messages)

# ========= 拼包 =========

def build_transitions_packet(sections_path:str, sentences_path:str, subp_json:str, llm_cfg:LLMConfig, model:str,mode:str,role_name:str)->Dict[str,Any]:
    sections=load_jsonl(sections_path)
    sentences=load_jsonl(sentences_path)
    subp=json.load(open(subp_json,"r",encoding="utf-8"))
    mode = mode.lower()
    role_name = role_name.lower()
    
    print(f"模式={mode}；角色={role_name}")
    logging.info(f"模式={mode}；角色={role_name}")
    subp_name=os.getenv("SUBP_NAME","Streams")
    # role_name=os.getenv("ROLE_NAME","sender")
    # mode=os.getenv("TRANSITION_MODE","mech").lower()  # frame | mech

    target=None
    for sp in subp.get("subprotocols",[]):
        if sp.get("name","").lower()==subp_name.lower(): target=sp; break
    if target:
        includes = sorted(set([sec.split(".")[0] for sec in target.get("boundaries",{}).get("includes",[])]))
        key_frames = [f for f in (sp.get("key_frames") or [])] if target else []
    else:
        includes = sorted(set([s.get("section","").split(".")[0] for s in sections if s.get("section")]))
        key_frames = list(STREAM_FRAMES)

    toc=[{"section":s["section"],"title":s.get("title",""),"anchor":s.get("anchor","")} for s in sections if s.get("section","").split(".")[0] in includes]

    buckets=collections.defaultdict(list)
    for s in sentences:
        sec=s.get("section",""); sec0=sec.split(".")[0]
        if sec0 not in includes: continue
        txt=s.get("text","")
        score=0.0
        if RFC2119.search(txt): score += 2.0
        if EVENT.search(txt): score += 1.0
        if mode=="frame":
            if not any(f in txt for f in key_frames): 
                continue
            score += 1.0
        else:  # mechanisms
            if not MECH_LEX.search(txt): 
                continue
            score += 1.0
        score /= max(10,len(txt))**0.5
        buckets[sec0].append((score,s))

    per = llm_cfg.MODEL_CONFIGS[model].doc_budget_tokens // max(1,len(includes))
    per = min(max(per, llm_cfg.MODEL_CONFIGS[model].min_sec_quota),
              llm_cfg.MODEL_CONFIGS[model].max_sec_quota)

    evidence=[]; seen=[]
    for sec0 in includes:
        items = sorted(buckets.get(sec0,[]), key=lambda x:-x[0])[:120]
        acc=0; block=[]
        for sc,s in items:
            txt=compress_clause(s["text"])
            h=simhash(txt)
            if any(ham(h,h0)<=8 for h0 in seen): continue
            toks=tk_len(txt, model)
            if acc+toks>per and block: break
            block.append({
                "type":"transition_hint",
                "text":txt,
                "source":{"section":s.get("section"),"anchor":s.get("anchor",""),
                          "para":s.get("para_id"),"sent_id":s.get("sent_id")}
            })
            seen.append(h); acc+=toks
        if block:
            title = next((x["title"] for x in sections if x["section"]==sec0),"")
            anch  = next((x.get("anchor","") for x in sections if x["section"]==sec0),"")
            evidence.append({"section":sec0,"title":title,"anchor":anch,"evidence":block})

    return {"subprotocol":subp_name,"role":role_name,"mode":mode,
            "key_frames":key_frames,
            "toc":toc,"evidence":evidence,
            "scope_hint":"转换抽取（帧/机制），规范+事件+帧附近片段优先"}

# ========= 提示 =========
SYS = """你是“协议规范编译器”。任务：仅依据提供的 RFC 证据，抽取目标子协议+角色在当前模式下的状态转换，并输出严格 JSON。
铁规：
1) 只用 evidence；每条 transition 与 invariant 都要 cites。
2) 模式=frame：事件必须来自 key_frames。
3) 模式=mechanisms：侧重 flow/congestion 的守卫、不变式；可引用相关帧作为事件。
4) 模态优先级：MUST/SHALL > SHOULD > MAY。
"""

DEV_FRAME = """【工作流】
A. 事件限定在 key_frames；生成 events[]。
B. 抽取 (from, event, to)，尽可能补充 guard（条件）与 action（动作列表），标注 normative_strength（MUST/SHOULD/MAY）。
C. 汇总 invariants（若规范句呈现不变条件）；无则给空数组。
D. 证据不足的写入 gaps。
E.cites 必须是字符串数组，仅包含来自 cites_pool 的 eid。禁止使用 section/anchor 文本；禁止自造 eid
【输出 JSON Schema】
{"subprotocol": str, "role": str, "mode":"frame_only",
 "events":[str],
 "transitions":[{"from":str,"event":str,"to":str,"guard":str,"action":[str],"normative_strength":str,"inferred":bool,"cites":[str]}],
 "invariants":[{"name":str,"expr":str,"cites":[str]}],
 "gaps":[{"ask":str,"wanted_sections":[str],"related_frames":[str]}]}
仅输出 JSON。
"""

DEV_MECH = """【工作流】
A. 聚焦机制触发（flow control/congestion 等），允许事件为机制事件名（如 FLOW_CREDIT_EXHAUSTED）或相关帧。
B. 抽取 (from,event,to) + guard/action + normative_strength；整理 invariants（尽可能用比较表达）。
C. 缺证据写入 gaps。
D.cites 必须是字符串数组，仅包含来自 cites_pool 的 eid。禁止使用 section/anchor 文本；禁止自造 eid
【输出 JSON Schema】
{"subprotocol": str, "role": str, "mode":"mechanisms",
 "events":[str],
 "transitions":[{"from":str,"event":str,"to":str,"guard":str,"action":[str],"normative_strength":str,"inferred":bool,"cites":[str]}],
 "invariants":[{"name":str,"expr":str,"cites":[str]}],
 "gaps":[{"ask":str,"wanted_sections":[str],"related_frames":[str]}]}
仅输出 JSON。
"""

def build_user_prompt(pkt:Dict[str,Any])->str:
    return ("任务：抽取状态转换；只输出 JSON。\n"
            "每条 transition/invariant 都要给 cites。\n\n"
            f"=== ContextPacket (transitions-{pkt['mode']}) ===\n{json.dumps(pkt,ensure_ascii=False,indent=2)}\n")

def shard_packet(packet:Dict[str,Any], base_tokens:int, hard_cap:int, model="gpt-4o")->List[Dict[str,Any]]:
    enc_budget = hard_cap - base_tokens - 256
    pj=json.dumps(packet, ensure_ascii=False, separators=(",",":"))
    if tk_len(pj, model) <= enc_budget: return [packet]
    shards=[]; cur={"subprotocol":packet["subprotocol"],"role":packet["role"],"mode":packet["mode"],
                    "key_frames":packet.get("key_frames",[]),"toc":[],"evidence":[],"scope_hint":packet["scope_hint"]}
    toc_map={t["section"]:t for t in packet["toc"]}
    def pkt_len(p): return tk_len(json.dumps(p, ensure_ascii=False, separators=(",",":")), model)
    for blk in packet["evidence"]:
        sec=blk["section"]
        tentative={**cur}; tentative["toc"]=cur["toc"]+[toc_map.get(sec,{"section":sec})]; tentative["evidence"]=cur["evidence"]+[blk]
        if pkt_len(tentative)<=enc_budget: cur=tentative
        else:
            if cur["evidence"]: shards.append(cur)
            cur={"subprotocol":packet["subprotocol"],"role":packet["role"],"mode":packet["mode"],
                 "key_frames":packet.get("key_frames",[]),"toc":[toc_map.get(sec,{"section":sec})],
                 "evidence":[blk],"scope_hint":packet["scope_hint"]}
    if cur["evidence"]: shards.append(cur)
    return shards

def merge_transitions_drafts(drafts: List[str]) -> Dict[str,Any]:
    subp=""; role=""; mode="frame_only"; events=set(); tr_map={}; inv_map={}; gaps=[]
    rank={"MUST":3,"SHALL":3,"REQUIRED":3,"SHOULD":2,"RECOMMENDED":2,"MAY":1,"OPTIONAL":1}
    def strong(a,b):
        ra=rank.get((a or "").upper(),0); rb=rank.get((b or "").upper(),0)
        return a if ra>=rb else b
    for dj in drafts:
        try:d=json.loads(dj)
        except: continue
        subp=subp or d.get("subprotocol","")
        role=role or d.get("role","")
        mode=d.get("mode",mode)
        for e in d.get("events",[]): events.add(e)
        for t in d.get("transitions",[]):
            key=( (t.get("from") or "").lower(), (t.get("event") or "").upper(), (t.get("to") or "").lower(), (t.get("guard") or "") )
            cur=tr_map.get(key, {"from":t.get("from"),"event":t.get("event"),"to":t.get("to"),
                                 "guard":t.get("guard"),"action":[], "normative_strength":None,"inferred":False,"cites":[]})
            cur["action"] = list(dict.fromkeys(cur["action"] + (t.get("action") or [])))
            cur["normative_strength"] = strong(cur["normative_strength"], t.get("normative_strength"))
            cur["inferred"] = cur["inferred"] or bool(t.get("inferred", False))
            cur["cites"] = list(dict.fromkeys(cur["cites"] + (t.get("cites") or [])))
            tr_map[key]=cur
        for iv in d.get("invariants",[]):
            k = (iv.get("name") or iv.get("expr") or "").strip()
            if not k: continue
            cur = inv_map.get(k, {"name":iv.get("name") or k, "expr":iv.get("expr") or "", "cites":[]})
            if len((iv.get("expr") or "")) > len(cur["expr"]): cur["expr"]=iv.get("expr")
            cur["cites"] = list(dict.fromkeys(cur["cites"] + (iv.get("cites") or [])))
            inv_map[k]=cur
        gaps += (d.get("gaps") or [])
    return {"subprotocol":subp,"role":role,"mode":mode if mode in ("mechanisms","frame_only") else ("mechanisms" if mode=="mechanisms" else "frame_only"),
            "events":sorted(events), "transitions": list(tr_map.values()),
            "invariants": list(inv_map.values()), "gaps": list({json.dumps(g,ensure_ascii=False,sort_keys=True):g for g in gaps}.values())}

def main():

    setup_logging()
    logging.info(f"启动时间：{now().strftime('%F %T %Z')}")

    sections = os.getenv("SECTIONS_JSONL","d:/ProtocolAutoModel-main/data/sections.jsonl")
    sentences = os.getenv("SENTENCES_JSONL","d:/ProtocolAutoModel-main/data/sentences.jsonl")
    subp_json = os.getenv("SUBP_JSON","./output/agents/subprotocol/QUIC_subprotocols_gpt-4o_developpromptchange.json")

    MODEL=os.getenv("MODEL_NAME","gpt-4o") # qwen2-5-72b-instruct / gpt-4o / claude-3-5-sonnet / deepseek-reasoner/ gpt-5

    MODE=os.getenv("TRANSITION_MODE","frame").lower() # frame | mech

    
    logging.info(f"模型={MODEL}；模式={MODE}")
    role_name=os.getenv("ROLE_NAME","receiver") # sender | receiver

    out_dir=os.path.join(os.getcwd(),"output","agents","transitions"); os.makedirs(out_dir, exist_ok=True)

    cfg = LLMConfig(api_key=os.getenv("OPENAI_API_KEY","sk-po4PH2gUa11tTr4733118e1f07944a64Ac1f2bC9E8158602"),
                    base_url=os.getenv("OPENAI_BASE_URL","https://api.gpt.ge/v1/"),
                    model_name=MODEL)
    HARD_CAP=cfg.hard_cap

    packet = build_transitions_packet(sections, sentences, subp_json, cfg, MODEL,MODE,role_name)

    sys_prompt = SYS
    dev_prompt = (DEV_FRAME if MODE=="receiver" else DEV_MECH)

    base_msgs=[{"role":"system","content":sys_prompt},
               {"role":"developer","content":dev_prompt},
               {"role":"user","content":"PLACEHOLDER"}]
    base_tokens = messages_tokens(base_msgs, MODEL)

    shards = shard_packet(packet, base_tokens, HARD_CAP, MODEL)

    llm = create_llm(MODEL, cfg)
    drafts=[]
    for i,p in enumerate(shards,1):
        user = build_user_prompt(p)
        msgs = llm.format_messages(sys_prompt, dev_prompt, user)
        tot = messages_tokens(msgs, MODEL)
        if tot>HARD_CAP:
            ev=list(p["evidence"])
            while ev and messages_tokens(llm.format_messages(sys_prompt, dev_prompt, build_user_prompt({**p,"evidence":ev})), MODEL)>HARD_CAP:
                ev = ev[:-1]
            p={**p,"evidence":ev}; msgs = llm.format_messages(sys_prompt, dev_prompt, build_user_prompt(p))
        rsp = llm.call(msgs, max_tokens=cfg.max_output_tokens)
        m=re.search(r'```json\s*(.*?)\s*```', rsp, re.S)
        text=(m.group(1) if m else rsp).strip()
        drafts.append(text)
        logging.info(f"[{i}] chars={len(text)}")

    merged = merge_transitions_drafts(drafts)
    out_path = os.path.join(out_dir, f"QUIC_{role_name}_transitions_{MODE}_{MODEL}_ftramechange.json")
    save_json(merged, out_path)

    print(json.dumps(merged, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
