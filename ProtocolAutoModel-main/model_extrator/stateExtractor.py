# -*- coding: utf-8 -*-
"""
States Extractor (Budgeted + Sharded + Merge)
- 输入：sections.jsonl / sentences.jsonl + subprotocols.json（限定 includes）+ 角色名称（ENV: ROLE_NAME）
- 输出：output/agents/QUIC_states_<MODEL>.json
"""
"""
20250909 问题记录:
-states 抽取结果不理想，存在遗漏和冗余,特别是receiver角色的提取


"""
import os, re, json, logging, sys, hashlib, collections
from typing import Any, Dict, List
from APIConfig import LLMConfig, create_llm
from datetime import datetime
# ========= 通用 =========
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

def now(tz="Asia/Shanghai"):
    import pytz
    return datetime.now(pytz.timezone(tz))

def load_jsonl(p): 
    with open(p,"r",encoding="utf-8") as f: 
        return [json.loads(l) for l in f if l.strip()]

def save_json(o,p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p,"w",encoding="utf-8") as f: json.dump(o,f,ensure_ascii=False,indent=2)

def setup_logging(log_file="states_prompt.log"):
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
EVENT = re.compile(r"\b(upon|when|if|then|receive[sd]?|send[sd]?|ack|blocked|timeout|expires?)\b", re.I)
STATE_LEX = re.compile(r"\b(state|ready|open|closed|half-closed|data\s+sent|data\s+received|idle|blocked)\b", re.I)

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

def compress_clause(text, keep=180):
    spans=[]
    for m in RFC2119.finditer(text): spans.append((m.start(),m.end()))
    for m in EVENT.finditer(text): spans.append((m.start(),m.end()))
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
def build_states_packet(sections_path:str, sentences_path:str, subp_json:str, llm_cfg:LLMConfig, model:str,role_name:str)->Dict[str,Any]:
    sections=load_jsonl(sections_path)
    sentences=load_jsonl(sentences_path)
    subp=json.load(open(subp_json,"r",encoding="utf-8"))
    role_name=role_name

    subp_name=os.getenv("SUBP_NAME","Streams")
    # role_name=os.getenv("ROLE_NAME","receiver")  # sender/receiver/client/server 等

    target=None
    for sp in subp.get("subprotocols",[]):
        if sp.get("name","").lower()==subp_name.lower(): target=sp; break
    if target: includes = sorted(set([sec.split(".")[0] for sec in target.get("boundaries",{}).get("includes",[])]))
    else: includes = sorted(set([s.get("section","").split(".")[0] for s in sections if s.get("section")]))

    toc=[{"section":s["section"],"title":s.get("title",""),"anchor":s.get("anchor","")} for s in sections if s.get("section","").split(".")[0] in includes]

    buckets=collections.defaultdict(list)
    for s in sentences:
        sec=s.get("section",""); sec0=sec.split(".")[0]
        if sec0 not in includes: continue
        txt=s.get("text","")
        if not (STATE_LEX.search(txt) or "state" in txt.lower() or EVENT.search(txt) or RFC2119.search(txt)):
            continue
        score = 0.0
        if RFC2119.search(txt): score += 2.0
        if EVENT.search(txt): score += 1.0
        if STATE_LEX.search(txt): score += 1.0
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
                "type":"state_hint",
                "text":txt,
                "source":{"section":s.get("section"),"anchor":s.get("anchor",""),
                          "para":s.get("para_id"),"sent_id":s.get("sent_id")}
            })
            seen.append(h); acc+=toks
        if block:
            title = next((x["title"] for x in sections if x["section"]==sec0),"")
            anch  = next((x.get("anchor","") for x in sections if x["section"]==sec0),"")
            evidence.append({"section":sec0,"title":title,"anchor":anch,"evidence":block})

    return {"subprotocol":subp_name, "role":role_name, "toc":toc, "evidence":evidence,
            "scope_hint":"状态名与进入/退出提示（规范/事件优先）"}

# ========= 提示 =========
SYSTEM_PROMPT = """你是“协议规范编译器”。任务：仅依据提供的 RFC 证据，抽取目标子协议+角色的状态集合（含别名）以及进入/退出提示，输出严格 JSON。
铁规：
1) 只能使用 evidence；每一条均需 cites。
2) 状态名保留大小写；可合并别名到 aliases。
3) 只输出证据中能支持的状态；不得臆造。

"""

DEVELOPER_PROMPT = """【工作流】
A. 在子协议+角色范围，枚举状态名；若同义/近义，合并为一个状态并写 aliases。
B. 每个状态收集 enter_hints（在何种事件/条件下进入）与 exit_hints（在何种事件/条件下退出），每条附 cites。
C. 不出现的状态不要输出。
D.cites 必须是字符串数组，仅包含来自 cites_pool 的 eid。禁止使用 section/anchor 文本；禁止自造 eid
【输出 JSON Schema】
{"subprotocol": str, "role": str, "states": [
  {"name": str, "aliases":[str], "enter_hints":[str], "exit_hints":[str], "cites":[str]}
]}
仅输出 JSON。
"""

def build_user_prompt(pkt:Dict[str,Any])->str:
    return ("任务：抽取目标子协议+角色的状态集合；补充进入/退出提示；只输出 JSON。\n"
            "每个字段都要给 cites。\n\n"
            f"=== ContextPacket (states) ===\n{json.dumps(pkt,ensure_ascii=False,indent=2)}\n")

def shard_packet(packet:Dict[str,Any], base_tokens:int, hard_cap:int, model="gpt-4o")->List[Dict[str,Any]]:
    enc_budget = hard_cap - base_tokens - 256
    pj=json.dumps(packet, ensure_ascii=False, separators=(",",":"))
    if tk_len(pj, model) <= enc_budget: return [packet]
    shards=[]; cur={"subprotocol":packet["subprotocol"],"role":packet["role"],"toc":[],"evidence":[],"scope_hint":packet["scope_hint"]}
    toc_map={t["section"]:t for t in packet["toc"]}
    def pkt_len(p): return tk_len(json.dumps(p, ensure_ascii=False, separators=(",",":")), model)
    for blk in packet["evidence"]:
        sec=blk["section"]
        tentative={**cur}; tentative["toc"]=cur["toc"]+[toc_map.get(sec,{"section":sec})]; tentative["evidence"]=cur["evidence"]+[blk]
        if pkt_len(tentative)<=enc_budget: cur=tentative
        else:
            if cur["evidence"]: shards.append(cur)
            cur={"subprotocol":packet["subprotocol"],"role":packet["role"],"toc":[toc_map.get(sec,{"section":sec})],
                 "evidence":[blk],"scope_hint":packet["scope_hint"]}
    if cur["evidence"]: shards.append(cur)
    return shards

def merge_states_drafts(drafts: List[str]) -> Dict[str,Any]:
    subp=""; role=""; mp={}
    for dj in drafts:
        try:d=json.loads(dj)
        except: continue
        subp=subp or d.get("subprotocol","")
        role=role or d.get("role","")
        for s in d.get("states",[]):
            name=(s.get("name") or "").strip()
            if not name: continue
            k=name.lower()
            cur=mp.get(k, {"name":name,"aliases":[],"enter_hints":[],"exit_hints":[],"cites":[]})
            cur["aliases"]=list(dict.fromkeys(cur["aliases"] + (s.get("aliases") or [])))
            cur["enter_hints"]=list(dict.fromkeys(cur["enter_hints"] + (s.get("enter_hints") or [])))
            cur["exit_hints"]=list(dict.fromkeys(cur["exit_hints"] + (s.get("exit_hints") or [])))
            cur["cites"]=list(dict.fromkeys(cur["cites"] + (s.get("cites") or [])))
            mp[k]=cur
    return {"subprotocol":subp, "role":role, "states": list(mp.values())}

def main():

    setup_logging()
    logging.info(f"启动时间：{now().strftime('%F %T %Z')}")

    sections = os.getenv("SECTIONS_JSONL","d:/ProtocolAutoModel-main/data/rfc9000_corpus/sections.jsonl")
    sentences = os.getenv("SENTENCES_JSONL","d:/ProtocolAutoModel-main/data/rfc9000_corpus/sentences.jsonl")
    subp_json = os.getenv("SUBP_JSON","./output/agents/subprotocol/QUIC_subprotocols_gpt-4o_developpromptchange.json")
    # qwen2-5-72b-instruct / gpt-4o / claude-3-5-sonnet / deepseek-reasoner
    MODEL=os.getenv("MODEL_NAME","deepseek-reasoner")   
    out_dir=os.path.join(os.getcwd(),"output","agents","states"); os.makedirs(out_dir, exist_ok=True)

    cfg = LLMConfig(api_key=os.getenv("OPENAI_API_KEY","sk-po4PH2gUa11tTr4733118e1f07944a64Ac1f2bC9E8158602"),
                    base_url=os.getenv("OPENAI_BASE_URL","https://api.gpt.ge/v1/"),
                    model_name=MODEL)
    HARD_CAP=cfg.hard_cap
    role_name = os.getenv("ROLE_NAME","receiver") # sender/receiver/client/server 等
    packet = build_states_packet(sections, sentences, subp_json, cfg, MODEL,role_name)

    base_msgs=[{"role":"system","content":SYSTEM_PROMPT},
               {"role":"developer","content":DEVELOPER_PROMPT},
               {"role":"user","content":"PLACEHOLDER"}]
    
    base_tokens = messages_tokens(base_msgs, MODEL)
    logging.info(f"模板基线 tokens ≈ {base_tokens}")

    shards = shard_packet(packet, base_tokens, HARD_CAP, MODEL)
    logging.info(f"分片数：{len(shards)}")

    llm = create_llm(MODEL, cfg)
    drafts=[]
    for i,p in enumerate(shards,1):
        user = build_user_prompt(p)
        msgs = llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT, user)
        tot = messages_tokens(msgs, MODEL)
        if tot>HARD_CAP:
            ev=list(p["evidence"])
            while ev and messages_tokens(llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT, build_user_prompt({**p,"evidence":ev})), MODEL)>HARD_CAP:
                ev = ev[:-1]
            p={**p,"evidence":ev}; msgs = llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT, build_user_prompt(p))
        
        logging.info(f"[Shard {i}] 发送 tokens≈{messages_tokens(msgs, MODEL)}")

        rsp = llm.call(msgs, max_tokens=cfg.max_output_tokens)
        m=re.search(r'```json\s*(.*?)\s*```', rsp, re.S)
        text=(m.group(1) if m else rsp).strip()
        drafts.append(text)
        
        logging.info(f"[Shard {i}] 收到结果 {len(text)} chars")
        
    merged = merge_states_drafts(drafts)

    
    out_path = os.path.join(out_dir, f"QUIC_{role_name}_states_{MODEL}_0911.json")
    save_json(merged, out_path)
    print(json.dumps(merged, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
