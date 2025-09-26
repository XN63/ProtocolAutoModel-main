# -*- coding: utf-8 -*-
"""
Subprotocol Extractor (Budgeted + Sharded + Merge)
- 证据只读（带 cites）
- 预算化拼包：限域/配额/条款级压缩/去重
- 自动分片：严格控 tokens
- 三段式 Prompt：system + developer(COT+Schema) + user
- 多分片 Map-Reduce 合并成唯一 JSON
- 引用一致性增强（Steps 2/3/4/5）：
  2) 受限引用池 cites_pool（eid 闭集）；
  3) 合并后 eid → 引用字符串回填；
  4) 固化源 HTML 与解析器版本元数据；
  5) 兜底校验与（轻量）自动修复。
"""
### 2025-09-12 by gpt-5-thinking
#### 调用模型：gpt-4o, sonnet, deepseek, qwen

import os, re, json, math, logging, sys, hashlib, collections
from datetime import datetime
from typing import Any, Dict, List, Tuple, Iterable, Set
import pytz

from APIConfig import LLMConfig, create_llm



# tokenizer
try:
    import tiktoken
    _ENC = None
    def get_enc(model: str = "gpt-4o"):
        global _ENC
        if _ENC is None:
            try:
                _ENC = tiktoken.encoding_for_model(model)
            except Exception:
                _ENC = tiktoken.get_encoding("cl100k_base")
        return _ENC
    def tk_len(s: str, model: str = "gpt-4o") -> int:
        return len(get_enc(model).encode(s))
except Exception:
    def tk_len(s: str, model: str = "gpt-4o") -> int:
        return max(1, len(s)//4)

# ----------------- 日志 -----------------

def setup_logging(log_file="subprotocol_prompt.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logging.info(f"日志初始化：{log_file}")

def now(tz="Asia/Shanghai"):
    return datetime.now(pytz.timezone(tz))

# ----------------- I/O -----------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ----------------- 预处理与工具 -----------------

RFC2119 = re.compile(r"\b(MUST(?:\s+NOT)?|SHOULD(?:\s+NOT)?|MAY|REQUIRED|RECOMMENDED|NOT\s+RECOMMENDED|OPTIONAL|SHALL(?:\s+NOT)?)\b")
EVENT = re.compile(r"\b(upon|when|whenever|if|then|receive[sd]?|send[sd]?|ack|fin|timeout|expires?)\b", re.I)
ALLCAPS = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")  # 全大写的标识符
PURPOSE = re.compile(r"\b(provides|purpose|used to|controls|responsible|intended to|aims to)\b", re.I)  # 目的句

STREAM_FRAMES = {"STREAM","RESET_STREAM","STOP_SENDING","STREAM_DATA_BLOCKED",
                 "DATA_BLOCKED","MAX_DATA","MAX_STREAM_DATA","MAX_STREAMS","STREAMS_BLOCKED"}

def simhash(text: str, ngram: int = 3) -> int:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    grams = [t[i:i+ngram] for i in range(max(1, len(t)-ngram+1))]
    v = [0]*64
    for g in grams:
        h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for b in range(64):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(64):
        if v[b] > 0:
            out |= (1 << b)
    return out

def hamdist(a: int, b: int) -> int:
    x = a ^ b; c = 0
    while x:
        x &= x-1; c += 1
    return c

def compress_clause(text: str, frames: set, keep_chars: int = 180) -> str:
    spans = []
    for m in RFC2119.finditer(text): spans.append((m.start(), m.end()))
    for m in EVENT.finditer(text): spans.append((m.start(), m.end()))
    for m in ALLCAPS.finditer(text):
        if m.group(1) in frames: spans.append((m.start(), m.end()))
    if not spans:
        return text[:keep_chars] + "…" if len(text) > keep_chars else text
    wins = []
    for s, e in spans: wins.append((max(0, s-40), min(len(text), e+40)))
    wins.sort()
    merged = []; cs, ce = wins[0]
    for s, e in wins[1:]:
        if s <= ce + 16: ce = max(ce, e)
        else: merged.append((cs, ce)); cs, ce = s, e
    merged.append((cs, ce))
    parts = [text[s:e] for s, e in merged]
    out = " … ".join(parts)
    return out[:keep_chars] + "…" if len(out) > keep_chars else out

# ----------------- 辅助：eid ↔ 引用字符串 -----------------

def make_ref_string(section: str, anchor: str, para: Any, sent_id: Any) -> str:
    anchor = anchor or ""
    return f"{section}#{anchor}/para{para}/sent-{sent_id}" if anchor else f"{section}/para-{para}/sent-{sent_id}"

def build_eid_maps(sentences_path: str) -> Tuple[Dict[str, str], Set[str]]:
    """从 sentences.jsonl 建 eid→引用字符串 映射。要求 sentences 中已有 eid 字段（用户已完成 Step 1）。"""
    eid2ref: Dict[str, str] = {}
    for s in load_jsonl(sentences_path):
        eid = s.get("eid")
        if not eid:
            # 兼容旧数据：尽量不报错，仅跳过
            continue
        ref = make_ref_string(s.get("section",""), s.get("anchor",""), s.get("para_id"), s.get("sent_id"))
        eid2ref[eid] = ref
    return eid2ref, set(eid2ref.keys())

# ----------------- 预算化拼包（子协议阶段专用） -----------------

def select_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in sections if any(s.get("section", "").startswith(p) for p in ("2","3","4","19"))]

def build_toc(sections) -> List[Dict[str, Any]]:
    return [{"section": s["section"], "title": s.get("title", ""), "anchor": s.get("anchor", ""), "order": s.get("order", 0)} for s in sections]

def build_frame_index(sentences, sections):
    out = []
    for sec in sections:
        if not sec["section"].startswith("19"): continue
        frames = set()
        for s in sentences:
            if s["section"].startswith(sec["section"]):
                for f in (s.get("entities", {}).get("frames") or []):
                    if f in STREAM_FRAMES: frames.add(f)
        if frames:
            out.append({"section": sec["section"], "title": sec["title"], "anchor": sec.get("anchor", ""), "frames": sorted(frames)})
    return out

def build_budgeted_packet(
    sections_path: str,
    sentences_path: str,
    llm_config: LLMConfig,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    # 获取模型特定的预算参数
    model_config = llm_config.MODEL_CONFIGS[model]
    doc_budget_tokens = model_config.doc_budget_tokens
    min_sec_quota = model_config.min_sec_quota
    max_sec_quota = model_config.max_sec_quota
    
    sections = load_jsonl(sections_path)
    sentences = load_jsonl(sentences_path)
    secs = select_sections(sections)  # 只要 §2/§3/§4/§19.*
    toc = build_toc(secs)  # 目录
    frame_idx = build_frame_index(sentences, secs)  # 流相关帧索引

    # 固化源 HTML / 解析器版本（Step 4）
    source_meta = {
        "rfc_html_url": os.getenv("RFC_HTML_URL", ""),
        "html_sha256": os.getenv("RFC_HTML_SHA256", ""),
        "parser_version": os.getenv("PARSER_VERSION", "v1"),
        "sentences_path": sentences_path,
        "sections_path": sections_path,
    }

    # 标段首
    para_first = set(); seen_para = {}
    for s in sentences:
        pid = s.get("para_id"); sid = s.get("sent_id")
        if pid and pid not in seen_para:
            seen_para[pid] = sid; para_first.add(sid)

    # 每节收集候选
    buckets = collections.defaultdict(list)
    for s in sentences:
        sec = s.get("section", "")
        if not any(sec.startswith(x["section"]) for x in secs):
            continue
        txt = s.get("text", ""); norm = s.get("normative", False); facet = s.get("facet", "")
        score = 0.0
        is_intro = (s.get("sent_id") in para_first or PURPOSE.search(txt) or facet.lower() in {"intro","purpose","streams","flow_control","state_machine","frames"})
        if is_intro: score += 3.0
        if norm and RFC2119.search(txt): score += 2.0
        ents = s.get("entities", {})
        if ents.get("frames"): score += 1.5
        if ents.get("states"): score += 1.0
        score /= max(10, len(txt)) ** 0.5
        buckets[sec.split(".")[0]].append((score, s))

    # 节配额
    uniq_secs = sorted({x["section"] for x in secs})
    per = doc_budget_tokens // max(1, len(uniq_secs))
    per = min(max(per, min_sec_quota), max_sec_quota)

    frames_set = set(STREAM_FRAMES)
    evidence = []  # 证据包
    used = 0
    seen_hash = []
    cites_pool_set: Set[str] = set()
    cites_pool: List[Dict[str, Any]] = []

    def add_to_pool(src: Dict[str, Any]):
        eid = src.get("eid")
        if eid and eid not in cites_pool_set:
            cites_pool_set.add(eid)
            cites_pool.append({
                "eid": eid,
                "section": src.get("section"),
                "anchor": src.get("anchor", ""),
                "para": src.get("para"),
                "sent_id": src.get("sent_id"),
            })

    for sec_id in uniq_secs:
        items = [(sc, s) for (sc, s) in buckets.get(sec_id, [])]
        items.sort(key=lambda x: -x[0])
        take = 0; acc = 0; sec_block = []
        for sc, s in items[:80]:
            txt = compress_clause(s.get("text", ""), frames_set, keep_chars=180)
            h = simhash(txt)
            if any(hamdist(h, h0) <= 8 for h0 in seen_hash):
                continue
            toks = tk_len(txt, model)
            if acc + toks > per and take > 0:
                break
            src = {
                "section": s.get("section"),
                "anchor": s.get("anchor", ""),
                "para": s.get("para_id"),
                "sent_id": s.get("sent_id"),
                "eid": s.get("eid"),  # Step 2: 证据携带 eid
            }
            sec_block.append({
                "type": "purpose" if (s.get("sent_id") in para_first or PURPOSE.search(s.get("text", ""))) else "normative",
                "text": txt,
                "source": src,
            })
            add_to_pool(src)
            seen_hash.append(h); acc += toks; take += 1
        if sec_block:
            title = next((x["title"] for x in secs if x["section"] == sec_id), "")
            anch = next((x.get("anchor", "") for x in secs if x["section"] == sec_id), "")
            evidence.append({"section": sec_id, "title": title, "anchor": anch, "evidence": sec_block})
            used += acc

    packet = {
        "toc": toc,
        "evidence": evidence,
        "frame_evidence": frame_idx,
        "cites_pool": cites_pool,   # Step 2: 受限引用池（闭集）
        "source_meta": source_meta,  # Step 4: 固化元数据
        "scope_hint": "RFC9000 §2/§3/§4/§19.*（导言/目的/规范句；流相关帧索引）",
    }
    return packet

# ----------------- 分片（严格控 tokens） -----------------

def messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o") -> int:
    return sum(4 + tk_len(m.get("content", ""), model) for m in messages)


def _packet_tokens(packet: Dict[str, Any], model: str) -> int:
    j = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    return tk_len(j, model)


def shard_packet(packet: Dict[str, Any], base_tokens: int, hard_cap: int, model: str = "gpt-4o") -> List[Dict[str, Any]]:
    """按 section 切成若干片，确保 base_tokens + packet_json_tokens <= hard_cap。
    同时针对每个分片，裁剪 cites_pool 仅保留该片 evidence 涵盖的 eid（Step 2）。
    """
    enc_budget = hard_cap - base_tokens - 256  # 给 JSON 语法/边距留余量
    if _packet_tokens(packet, model) <= enc_budget:
        # 也要缩小 cites_pool 到这一包实际使用的 eids
        all_eids = {ev["source"]["eid"] for blk in packet.get("evidence", []) for ev in blk.get("evidence", []) if ev.get("source", {}).get("eid")}
        packet = dict(packet)
        packet["cites_pool"] = [p for p in packet.get("cites_pool", []) if p.get("eid") in all_eids]
        return [packet]

    shards = []
    toc_map = {t["section"]: t for t in packet["toc"]}
    frame_by_sec = collections.defaultdict(list)
    for f in packet.get("frame_evidence", []):
        frame_by_sec[f["section"]].append(f)

    def build_shard(sec_set: Set[str], evidence_blocks: List[Dict[str, Any]]):
        eids = {ev["source"].get("eid") for blk in evidence_blocks for ev in blk.get("evidence", []) if ev.get("source")}
        cites_pool = [p for p in packet.get("cites_pool", []) if p.get("eid") in eids]
        return {
            "toc": [toc_map[s] for s in sec_set if s in toc_map],
            "evidence": evidence_blocks,
            "frame_evidence": [x for s in sec_set for x in frame_by_sec.get(s, [])],
            "cites_pool": cites_pool,
            "source_meta": packet.get("source_meta", {}),
            "scope_hint": packet["scope_hint"],
        }

    cur_secs: Set[str] = set(); cur_blocks: List[Dict[str, Any]] = []
    for blk in packet["evidence"]:
        tentative_secs = cur_secs | {blk["section"]}
        tentative_blocks = cur_blocks + [blk]
        tentative = build_shard(tentative_secs, tentative_blocks)
        if _packet_tokens(tentative, model) <= enc_budget:
            cur_secs, cur_blocks = tentative_secs, tentative_blocks
        else:
            if cur_blocks:
                shards.append(build_shard(cur_secs, cur_blocks))
            cur_secs, cur_blocks = {blk["section"]}, [blk]
    if cur_blocks:
        shards.append(build_shard(cur_secs, cur_blocks))
    return shards

# ----------------- 三段式 Prompt（子协议阶段） -----------------

SYSTEM_PROMPT = """你是“协议规范编译器”。任务：仅依据提供的 RFC 证据（含章节号与锚点），
把 QUIC (RFC 9000) 的主要子协议划分出来，并给出用途、关键帧、依赖与边界。
铁规：
1) 仅使用输入 evidence；禁止外部知识或猜测。
2) 每条结论必须给 cites（引用 eid，来自 cites_pool）。
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
F. cites 必须是**字符串数组**，且只能取自 cites_pool.eid。禁止使用 section/anchor 文本；禁止自造 eid。
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


def build_user_prompt(context_packet: Dict[str, Any]) -> str:
    j = json.dumps(context_packet, ensure_ascii=False, indent=2)
    return (
        "任务：在 RFC 证据范围内，抽取‘主要子协议/主要部分’用于后续选择建模范围。\n"
        "只输出符合上方 JSON Schema 的 JSON。每条字段都要给 cites，且 cites 只能是 cites_pool 中的 eid（字符串）。\n\n"
        f"=== ContextPacket (subprotocols) ===\n{j}\n"
    )

# ----------------- 合并 / 验证 / 回填 -----------------

def merge_drafts(drafts: List[str]) -> Dict[str, Any]:
    out = {"subprotocols": [], "open_questions": []}
    by_name: Dict[str, Dict[str, Any]] = {}
    for dj in drafts:
        try:
            d = json.loads(dj)
        except Exception:
            continue
        for sp in d.get("subprotocols", []):
            name = (sp.get("name") or "").strip()
            if not name:
                continue
            k = name.lower()
            cur = by_name.get(k, {"name": name, "purpose": "", "key_frames": [], "depends_on": [], "boundaries": {"includes": [], "excludes": []}, "cites": []})
            if len(sp.get("purpose", "")) > len(cur["purpose"]):
                cur["purpose"] = sp.get("purpose", "")
            cur["key_frames"] = sorted(list(set(cur["key_frames"] + (sp.get("key_frames") or []))))
            cur["depends_on"] = sorted(list(set(cur["depends_on"] + (sp.get("depends_on") or []))))
            b = sp.get("boundaries", {}) or {}
            cur["boundaries"]["includes"] = sorted(list(set(cur["boundaries"]["includes"] + (b.get("includes") or []))))
            cur["boundaries"]["excludes"] = sorted(list(set(cur["boundaries"]["excludes"] + (b.get("excludes") or []))))
            cur["cites"] = sorted(list(set(cur["cites"] + (sp.get("cites") or []))))
            by_name[k] = cur
        out["open_questions"] += d.get("open_questions", [])
    # open_questions 去重
    seen = set(); oq = []
    for q in out["open_questions"]:
        k = json.dumps(q, ensure_ascii=False, sort_keys=True)
        if k in seen: continue
        seen.add(k); oq.append(q)
    out["open_questions"] = oq
    out["subprotocols"] = list(by_name.values())
    return out


def iter_all_items(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        if "cites" in obj and isinstance(obj["cites"], list):
            yield obj
        for v in obj.values():
            yield from iter_all_items(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_all_items(x)


def validate_and_fix_cites(obj: Dict[str, Any], valid_eids: Set[str]) -> Dict[str, Any]:
    """兜底校验（Step 5）：
    - 丢弃不在 valid_eids 中的自造/过期 eid；
    - 统计 unknown/empty；
    （轻量修复：本实现仅过滤，不做相似度对齐；若需可在此处扩展基于 text 的回补。）
    """
    unknown = 0; emptied = 0; total = 0
    for item in iter_all_items(obj):
        cites = item.get("cites", [])
        total += len(cites)
        ok = [e for e in cites if e in valid_eids]
        unknown += (len(cites) - len(ok))
        if not ok:
            emptied += 1
        item["cites"] = ok
    obj.setdefault("_cites_validation", {})
    obj["_cites_validation"].update({
        "total_cites": total,
        "unknown_eids": unknown,
        "empty_cites_items": emptied,
    })
    if unknown:
        logging.warning(f"cites 校验：发现 {unknown} 个未知 eid，被过滤；{emptied} 个条目变为空。")
    return obj


def backfill_cites(obj: Dict[str, Any], eid2ref: Dict[str, str]) -> Dict[str, Any]:
    def _map_list(lst: List[str]) -> List[str]:
        out = []
        for e in lst:
            r = eid2ref.get(e)
            if r: out.append(r)
        return out
    for item in iter_all_items(obj):
        item["cites"] = _map_list(item.get("cites", []))
    return obj

# ----------------- 主流程 -----------------

def main():
    setup_logging()
    logging.info(f"启动时间：{now().strftime('%F %T %Z')}")

    # === 路径与模型参数 ===
    sections_path = os.getenv("SECTIONS_JSONL", "d:/ProtocolAutoModel-main/data/rfc_9000_eid_corpus/sections.jsonl")
    sentences_path = os.getenv("SENTENCES_JSONL", "d:/ProtocolAutoModel-main/data/rfc_9000_eid_corpus/sentences.jsonl")
    out_dir = os.path.join(os.getcwd(), "output", "agents","subprotocol")
    os.makedirs(out_dir, exist_ok=True)

    MODEL = os.getenv("MODEL_NAME", "gpt-4o")

    # === 统一的 API 配置 ===
    llm_config = LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY", "sk-po4PH2gUa11tTr4733118e1f07944a64Ac1f2bC9E8158602"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.gpt.ge/v1/"),
        model_name=MODEL,
    )

    # 从配置中获取 token 限制
    MAX_INPUT = llm_config.max_input_tokens
    RESERVE = llm_config.max_output_tokens
    HARD_CAP = llm_config.hard_cap

    logging.info(f"MODEL={MODEL}, MAX_INPUT={MAX_INPUT}, HARD_CAP={HARD_CAP}, MAX_OUTPUT_TOKENS={RESERVE}")

    # === 1) 预算化拼包（单包） ===
    packet = build_budgeted_packet(
        sections_path=sections_path,
        sentences_path=sentences_path,
        llm_config=llm_config,
        model=MODEL,
    )

    # 估算模板自身 token（system+developer）
    base_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": "PLACEHOLDER"},
    ]
    base_tokens = messages_tokens(base_msgs, MODEL)
    logging.info(f"模板基线 tokens ≈ {base_tokens}")

    # === 2) 分片：严格控 tokens + cites_pool 限域 ===
    shards = shard_packet(packet, base_tokens, HARD_CAP, MODEL)
    logging.info(f"分片数：{len(shards)}")

    # === 创建 LLM 实例 ===
    llm = create_llm(MODEL, llm_config)

    # === 3) 逐片调用 ===
    drafts: List[str] = []
    union_pool_eids: Set[str] = set()
    for i, pkt in enumerate(shards, start=1):
        # 每片都带 cites_pool（eid 闭集）
        user_prompt = build_user_prompt(pkt)
        messages = llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT, user_prompt)
        tot = messages_tokens(messages, MODEL)
        # 仍超则裁剪 evidence 数组直到满足
        if tot > HARD_CAP:
            logging.warning(f"[Shard {i}] 超预算，启动二次裁剪 evidence…")
            safe = dict(pkt)
            ev = list(safe.get("evidence", []))
            while ev and messages_tokens(
                llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT, build_user_prompt({**safe, "evidence": ev})),
                MODEL,
            ) > HARD_CAP:
                ev = ev[:-1]  # 每次去掉一个 section block
            safe["evidence"] = ev
            # 同步裁剪后的 cites_pool
            eids = {v["source"].get("eid") for b in ev for v in b.get("evidence", []) if v.get("source")}
            safe["cites_pool"] = [p for p in safe.get("cites_pool", []) if p.get("eid") in eids]
            user_prompt = build_user_prompt(safe)
            messages = llm.format_messages(SYSTEM_PROMPT, DEVELOPER_PROMPT, user_prompt)
            tot = messages_tokens(messages, MODEL)
        logging.info(f"[Shard {i}] 发送 tokens≈{tot}")

        # 扩展日志：记录送入模型的prompt内容
        logging.info("====prompt====")
        logging.info(f"System Prompt:\n{SYSTEM_PROMPT}")
        logging.info("----")
        logging.info(f"Developer Prompt:\n{DEVELOPER_PROMPT}")
        logging.info("----")
        logging.info(f"User Prompt:\n{user_prompt}")
        logging.info("----")
        logging.info(f"Packet Data:\n{json.dumps(pkt, ensure_ascii=False, indent=2)}")
        logging.info("====end prompt====")

        rsp = llm.call(messages, max_tokens=llm.config.max_output_tokens)
        m = re.search(r'```json\s*(.*?)\s*```', rsp, re.DOTALL)
        text = m.group(1).strip() if m else rsp.strip()
        drafts.append(text)
        # 收集本片的合法 pool eids（用于后续校验）
        for p in pkt.get("cites_pool", []):
            if p.get("eid"):
                union_pool_eids.add(p["eid"])
        logging.info(f"[Shard {i}] 收到结果 {len(text)} chars")

    # === 4) 合并草案 ===
    if not drafts:
        logging.error("没有收到任何草案，无法合并。请检查前面的日志以了解详情。")
        return
    merged = drafts[0] if len(drafts) == 1 else json.dumps(merge_drafts(drafts), ensure_ascii=False)
    merged_obj = json.loads(merged)

    # === 5) 兜底校验 cites（限制为 union_pool_eids） ===
    merged_obj = validate_and_fix_cites(merged_obj, union_pool_eids or set())

    # === 6) 回填 eid → 引用字符串（基于 sentences.jsonl 构建 eid2ref） ===
    eid2ref, valid_all_eids = build_eid_maps(sentences_path)
    merged_obj = backfill_cites(merged_obj, eid2ref)

    # 输出
    out_path = os.path.join(out_dir, f"QUIC_subprotocols_{MODEL}_make_eid.json")
    save_json(merged_obj, out_path)
    print("===== 子协议提取（合并+校验+回填后）=====")
    print(json.dumps(merged_obj, ensure_ascii=False, indent=2))
    logging.info(f"已保存：{out_path}")


if __name__ == "__main__":
    main()
