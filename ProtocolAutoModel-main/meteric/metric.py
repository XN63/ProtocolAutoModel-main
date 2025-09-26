# metrics.py
# 评测四个任务：子协议 / 角色 / 状态 / 转换（frame & mechanisms）
# 用法示例：
#   python metrics.py \
#     --pred_subproto ./output/agents/QUIC_subprotocols_gpt-4o.json \
#     --pred_roles ./output/agents/QUIC_roles_gpt-4o.json \
#     --pred_states ./output/agents/QUIC_states_gpt-4o.json \
#     --pred_trans_frame ./output/agents/QUIC_transitions_frame_gpt-4o.json \
#     --pred_trans_mech  ./output/agents/QUIC_transitions_mech_gpt-4o.json \
#     --gold_dir ./gold \
#     --sentences_jsonl /path/to/sentences.jsonl
#
# gold_xxx.json 的期望结构请看各 evaluate_* 函数顶部的注释。

from __future__ import annotations
import json, re, argparse, os, sys, math, difflib
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

# ---------- 小工具 ----------
def load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out=[]
    if not path or not os.path.exists(path): return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

def jaccard(a: List[str], b: List[str]) -> float:
    A={x.strip().lower() for x in a if x}
    B={x.strip().lower() for x in b if x}
    if not A and not B: return 1.0
    return len(A & B)/max(1,len(A|B))

def prf1(tp:int, pN:int, gN:int) -> Dict[str,float]:
    prec = tp/max(1,pN); rec = tp/max(1,gN)
    f1 = (0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec))
    return {"precision":prec,"recall":rec,"f1":f1}

# ---------- RFC 信号（用于 cites 紧致度） ----------
RFC2119 = re.compile(r"\b(MUST(?:\s+NOT)?|SHOULD(?:\s+NOT)?|MAY|REQUIRED|RECOMMENDED|NOT\s+RECOMMENDED|OPTIONAL|SHALL(?:\s+NOT)?)\b")
EVENT   = re.compile(r"\b(upon|when|whenever|if|then|receive[sd]?|send[sd]?|ack|fin|timeout|expires?)\b", re.I)
ALLCAPS = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")

# ---------- cites 解析 ----------
CITE_PAT = re.compile(
    r"^(?P<section>\d+(?:\.\d+)*)"
    r"(?:#(?P<anchor>[^/]+))?"
    r"(?:/para(?P<para>\d+))?"
    r"(?:/sent(?P<sent>\d+))?$"
)

def parse_cite(s: str) -> Dict[str, Any]:
    """解析 cite 字符串。支持 '2.1#anchor/para12/sent3' 或仅有 'para/sent'，或 eid"""
    s = (s or "").strip()
    if not s: return {}
    # eid（十六进制 16 位或以上）
    if re.fullmatch(r"[0-9a-fA-F]{12,64}", s):
        return {"eid": s}
    m = CITE_PAT.match(s)
    if not m:
        # 尝试只含 para/sent 的形式
        m2 = re.search(r"para(\d+).*?sent(\d+)", s)
        if m2:
            return {"para": int(m2.group(1)), "sent_id": int(m2.group(2))}
        return {}
    d = m.groupdict()
    out={}
    if d.get("section"): out["section"] = d["section"]
    if d.get("anchor"): out["anchor"] = d["anchor"]
    if d.get("para"):   out["para"]   = int(d["para"])
    if d.get("sent"):   out["sent_id"]= int(d["sent"])
    return out

def index_sentences_by_keys(sentences: List[Dict[str,Any]]):
    """建立多个键的倒排，便于验证 cite"""
    by_full = {}    # (section, anchor, para, sent_id) -> text
    by_ps   = {}    # (para, sent_id) -> text
    by_eid  = {}    # eid -> text  （如果 sentences 里有 eid 字段）
    for s in sentences:
        sec = s.get("section") or ""
        anc = s.get("anchor") or ""
        para = s.get("para_id")
        sid  = s.get("sent_id")
        txt  = s.get("text") or ""
        keyf = (sec, anc, para, sid)
        by_full[keyf] = txt
        if para is not None and sid is not None:
            by_ps[(para, sid)] = txt
        if "eid" in s:
            by_eid[s["eid"]] = txt
    return by_full, by_ps, by_eid

def fetch_text_for_cite(cite: str, idx_full, idx_ps, idx_eid) -> Optional[str]:
    d = parse_cite(cite)
    if not d: return None
    if "eid" in d:
        return idx_eid.get(d["eid"])
    keyf = (d.get("section",""), d.get("anchor",""), d.get("para"), d.get("sent_id"))
    if keyf in idx_full:
        return idx_full[keyf]
    if d.get("para") is not None and d.get("sent_id") is not None:
        return idx_ps.get((d["para"], d["sent_id"]))
    return None

# ---------- 证据层指标 ----------
def cites_coverage(items: List[Dict[str,Any]]) -> Dict[str,float]:
    """对象级是否至少有一条 cite"""
    if not items: return {"coverage": 0.0, "n": 0}
    hit = sum(1 for it in items if (it.get("cites") and len(it["cites"])>0))
    return {"coverage": hit/len(items), "n": len(items)}

def cites_validity(items: List[Dict[str,Any]], idx_full, idx_ps, idx_eid) -> Dict[str,float]:
    """cites 是否能在 sentences.jsonl 对上"""
    total=0; ok=0
    for it in items:
        for c in (it.get("cites") or []):
            total+=1
            if fetch_text_for_cite(c, idx_full, idx_ps, idx_eid) is not None:
                ok+=1
    rate = (ok/total) if total>0 else 1.0
    return {"valid_rate": rate, "total": total}

def cites_tightness(items: List[Dict[str,Any]], idx_full, idx_ps, idx_eid) -> Dict[str,float]:
    """
    取每个对象的第一条 cite 对应句，检查是否包含 RFC2119 或事件/帧 ALLCAPS 等信号。
    """
    if not items: return {"tight_rate": 0.0, "n": 0}
    hit=0; n=0
    for it in items:
        cites = it.get("cites") or []
        if not cites: continue
        txt = fetch_text_for_cite(cites[0], idx_full, idx_ps, idx_eid)
        if not txt: continue
        n+=1
        if RFC2119.search(txt) or EVENT.search(txt) or ALLCAPS.search(txt):
            hit+=1
    return {"tight_rate": (hit/max(1,n)), "n": n}

def cites_overlap_with_gold(pred_items: List[Dict[str,Any]], gold_items: List[Dict[str,Any]],
                            key: str) -> Dict[str,float]:
    """
    对按 key 匹配的对象（如 name 或 (from,event,to) 文本串），
    统计 cites 集合与 gold 的重叠情况（至少一条重叠的比例）。
    """
    gmap = {}
    for g in gold_items:
        k = (g.get(key) or "").strip().lower()
        if not k: continue
        gmap[k] = set(g.get("cites") or [])
    if not gmap: return {"overlap_rate": 0.0, "n": 0}

    n=0; hit=0
    for p in pred_items:
        k = (p.get(key) or "").strip().lower()
        if not k or k not in gmap: continue
        n+=1
        if gmap[k] & set(p.get("cites") or []):
            hit+=1
    return {"overlap_rate": (hit/max(1,n)), "n": n}

# ---------- 语义层：各任务 ----------
def eval_subprotocols(pred: Dict[str,Any], gold: Dict[str,Any]) -> Dict[str,Any]:
    """
    gold_subprotocols.json 结构示例：
    {
      "subprotocols": [
        {"name":"Streams",
         "key_frames":["STREAM","RESET_STREAM",...],
         "boundaries":{"includes":["2","3","4","19"]}}
      ]
    }
    """
    p_list = pred.get("subprotocols", [])
    g_list = gold.get("subprotocols", [])
    # 名称 Jaccard
    pj=[x.get("name","") for x in p_list]; gj=[x.get("name","") for x in g_list]
    name_j = jaccard(pj, gj)

    # key_frames （对每个同名子协议）
    kf_stats=[]
    for g in g_list:
        gname = (g.get("name") or "").strip().lower()
        gkf = set([x.upper() for x in g.get("key_frames", [])])
        pkf = set()
        for p in p_list:
            if (p.get("name") or "").strip().lower()==gname:
                pkf |= set([x.upper() for x in (p.get("key_frames") or [])])
        tp=len(gkf & pkf)
        rec=tp/max(1,len(gkf)); prec=tp/max(1,len(pkf))
        f1=(0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec))
        kf_stats.append({"name": g.get("name",""), "precision":prec, "recall":rec, "f1":f1,
                         "gold":len(gkf), "pred":len(pkf), "tp":tp})

    # boundaries.includes P/R（对每个同名）
    b_stats=[]
    for g in g_list:
        gname=(g.get("name") or "").strip().lower()
        gincl=set((g.get("boundaries") or {}).get("includes", []))
        pincl=set()
        for p in p_list:
            if (p.get("name") or "").strip().lower()==gname:
                pincl |= set((p.get("boundaries") or {}).get("includes", []))
        tp=len(gincl & pincl); prec=tp/max(1,len(pincl)); rec=tp/max(1,len(gincl))
        f1=(0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec))
        b_stats.append({"name": g.get("name",""), "precision":prec, "recall":rec, "f1":f1,
                        "gold":len(gincl), "pred":len(pincl), "tp":tp})

    return {"name_jaccard": name_j, "key_frames": kf_stats, "boundaries": b_stats}

def eval_roles(pred: Dict[str,Any], gold: Dict[str,Any]) -> Dict[str,Any]:
    """
    gold_roles.json：
    {"subprotocol":"Streams",
     "roles":[{"name":"StreamSender"}, {"name":"StreamReceiver"}]}
    """
    pr = [r.get("name","") for r in pred.get("roles",[])]
    gr = [r.get("name","") for r in gold.get("roles",[])]
    return {"roles_jaccard": jaccard(pr, gr), "pred_N":len(pr), "gold_N":len(gr)}

def eval_states(pred: Dict[str,Any], gold: Dict[str,Any]) -> Dict[str,Any]:
    """
    gold_states.json：
    {"subprotocol":"Streams","role":"StreamSender",
     "states":[{"name":"Ready"},{"name":"Send"},{"name":"Data Sent"},{"name":"Reset Sent"},{"name":"Closed"}]}
    （如按角色拆多份 gold，可分别调用后加权平均）
    """
    ps = [s.get("name","") for s in pred.get("states",[])]
    gs = [s.get("name","") for s in gold.get("states",[])]
    return {"states_jaccard": jaccard(ps, gs), "pred_N":len(ps), "gold_N":len(gs)}

# 转换匹配：严格/宽松
def canon_event_loose(ev: str) -> str:
    ev=(ev or "").strip().upper()
    if not ev: return ev
    # 只取帧名的主干（去掉可能的后缀、空格、: 等）
    ev = re.split(r"[:/#\s]", ev)[0]
    return ev

def triples_from_trans(lst: List[Dict[str,Any]], loose=False) -> set:
    out=set()
    for t in lst or []:
        f=(t.get("from") or "").strip().lower()
        e=(t.get("event") or "").strip().upper()
        to=(t.get("to") or "").strip().lower()
        if loose: e=canon_event_loose(e)
        if f and e and to: out.add((f,e,to))
    return out

def eval_transitions(pred: Dict[str,Any], gold: Dict[str,Any]) -> Dict[str,Any]:
    """
    gold_transitions_*.json:
    {"subprotocol":"Streams","role":"StreamSender",
     "transitions":[{"from":"Open","event":"STREAM","to":"Data Sent"},
                    {"from":"Open","event":"RESET_STREAM","to":"Reset Sent"}, ...]}
    """
    P  = triples_from_trans(pred.get("transitions", []), loose=False)
    P2 = triples_from_trans(pred.get("transitions", []), loose=True)
    G  = triples_from_trans(gold.get("transitions", []), loose=False)
    G2 = triples_from_trans(gold.get("transitions", []), loose=True)

    strict = prf1(len(P & G), len(P), len(G))
    loose  = prf1(len(P2 & G2), len(P2), len(G2))

    return {"strict": strict, "loose": loose,
            "pred_N": len(P), "gold_N": len(G)}

# FSM 健康检查
def fsm_health(pred: Dict[str,Any]) -> Dict[str,Any]:
    states=set()
    indeg=Counter(); outdeg=Counter()
    fanout=defaultdict(set)
    for t in pred.get("transitions", []):
        s=(t.get("from") or "").strip()
        e=(t.get("event") or "").strip().upper()
        d=(t.get("to") or "").strip()
        if not (s and e and d): continue
        states|={s,d}; indeg[d]+=1; outdeg[s]+=1
        fanout[(s,e)].add(d)
    unreachable=sorted([s for s in states if indeg[s]==0])
    dead=sorted([s for s in states if outdeg[s]==0 and s.lower() not in ("closed","reset","terminated")])
    conflicts=sorted([f"{k[0]} + {k[1]} -> {sorted(v)}" for k,v in fanout.items() if len(v)>1])
    return {"unreachable": unreachable, "dead": dead, "conflicts": conflicts}

# ---------- 结构层（Schema 简验，若未装 jsonschema 则跳过） ----------
def schema_valid(obj: Dict[str,Any], stage:str) -> bool:
    try:
        import jsonschema
    except Exception:
        return True
    SCHEMAS = {
        "subprotocols":{
            "type":"object","required":["subprotocols"],
            "properties":{"subprotocols":{"type":"array"}}
        },
        "roles":{
            "type":"object","required":["roles"],
            "properties":{"roles":{"type":"array"}}
        },
        "states":{
            "type":"object","required":["states"],
            "properties":{"states":{"type":"array"}}
        },
        "transitions":{
            "type":"object","required":["transitions"],
            "properties":{"transitions":{"type":"array"}}
        }
    }
    try:
        jsonschema.validate(obj, SCHEMAS.get(stage, {"type":"object"}))
        return True
    except Exception:
        return False

# ---------- 汇总打印 ----------
def pretty_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_subproto", required=True)
    ap.add_argument("--pred_roles", required=True)
    ap.add_argument("--pred_states", required=True)
    ap.add_argument("--pred_trans_frame", required=False, default="")
    ap.add_argument("--pred_trans_mech", required=False, default="")
    ap.add_argument("--gold_dir", required=True)
    ap.add_argument("--sentences_jsonl", required=False, default="")
    args=ap.parse_args()

    # 载入
    pred_sub = load_json(args.pred_subproto)
    pred_rol = load_json(args.pred_roles)
    pred_sta = load_json(args.pred_states)
    pred_tf  = load_json(args.pred_trans_frame) if args.pred_trans_frame else {}
    pred_tm  = load_json(args.pred_trans_mech)  if args.pred_trans_mech  else {}

    gold_sub = load_json(os.path.join(args.gold_dir, "gold_subprotocols.json"))
    gold_rol = load_json(os.path.join(args.gold_dir, "gold_roles.json"))
    gold_sta = load_json(os.path.join(args.gold_dir, "gold_states.json"))
    gold_tf  = load_json(os.path.join(args.gold_dir, "gold_transitions_frame.json"))
    gold_tm  = load_json(os.path.join(args.gold_dir, "gold_transitions_mech.json"))

    sentences = load_jsonl(args.sentences_jsonl) if args.sentences_jsonl else []
    idx_full, idx_ps, idx_eid = index_sentences_by_keys(sentences)

    # 结构层
    struct = {
        "subprotocols_schema_ok": schema_valid(pred_sub, "subprotocols"),
        "roles_schema_ok":        schema_valid(pred_rol, "roles"),
        "states_schema_ok":       schema_valid(pred_sta, "states"),
        "trans_frame_schema_ok":  schema_valid(pred_tf,  "transitions") if pred_tf else None,
        "trans_mech_schema_ok":   schema_valid(pred_tm,  "transitions") if pred_tm else None
    }

    # 证据层（按对象粒度）
    # 子协议对象=每个子协议；角色对象=每个角色；状态对象=每个状态；转换对象=每条 transition
    sp_objs = [{"cites": sp.get("cites", [])} for sp in pred_sub.get("subprotocols", [])]
    rl_objs = [{"cites": r.get("cites", [])} for r in pred_rol.get("roles", [])]
    st_objs = [{"cites": s.get("cites", [])} for s in pred_sta.get("states", [])]
    tf_objs = [{"cites": t.get("cites", [])} for t in pred_tf.get("transitions", [])] if pred_tf else []
    tm_objs = [{"cites": t.get("cites", [])} for t in pred_tm.get("transitions", [])] if pred_tm else []

    evid = {
        "subprotocols": {
            "coverage": cites_coverage(sp_objs),
            "validity": cites_validity(sp_objs, idx_full, idx_ps, idx_eid),
            "tightness": cites_tightness(sp_objs, idx_full, idx_ps, idx_eid),
        },
        "roles": {
            "coverage": cites_coverage(rl_objs),
            "validity": cites_validity(rl_objs, idx_full, idx_ps, idx_eid),
            "tightness": cites_tightness(rl_objs, idx_full, idx_ps, idx_eid),
        },
        "states": {
            "coverage": cites_coverage(st_objs),
            "validity": cites_validity(st_objs, idx_full, idx_ps, idx_eid),
            "tightness": cites_tightness(st_objs, idx_full, idx_ps, idx_eid),
        },
        "transitions_frame": (None if not tf_objs else {
            "coverage": cites_coverage(tf_objs),
            "validity": cites_validity(tf_objs, idx_full, idx_ps, idx_eid),
            "tightness": cites_tightness(tf_objs, idx_full, idx_ps, idx_eid),
        }),
        "transitions_mech": (None if not tm_objs else {
            "coverage": cites_coverage(tm_objs),
            "validity": cites_validity(tm_objs, idx_full, idx_ps, idx_eid),
            "tightness": cites_tightness(tm_objs, idx_full, idx_ps, idx_eid),
        })
    }

    # 语义层
    sub_sem = eval_subprotocols(pred_sub, gold_sub)
    rol_sem = eval_roles(pred_rol, gold_rol)
    sta_sem = eval_states(pred_sta, gold_sta)
    tf_sem  = (eval_transitions(pred_tf, gold_tf) if pred_tf and gold_tf else None)
    tm_sem  = (eval_transitions(pred_tm, gold_tm) if pred_tm and gold_tm else None)

    # FSM 健康（转换）
    tf_fsm = (fsm_health(pred_tf) if pred_tf else None)
    tm_fsm = (fsm_health(pred_tm) if pred_tm else None)

    report = {
        "STRUCTURE": struct,
        "EVIDENCE": evid,
        "SEMANTICS": {
            "subprotocols": sub_sem,
            "roles": rol_sem,
            "states": sta_sem,
            "transitions_frame": tf_sem,
            "transitions_mech":  tm_sem
        },
        "FSM_HEALTH": {
            "frame": tf_fsm,
            "mechanisms": tm_fsm
        }
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
