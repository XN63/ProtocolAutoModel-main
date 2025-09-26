# -*- coding: utf-8 -*-

"""
Build QUIC RFC9000 corpus (Prefer HTML, local file supported):
- Parse RFC9000 HTML from --html_path (preferred) or --url
- Extract section tree, paragraph blocks, sentence-level micro-chunks
- Auto-tag: facet / normative / modalities / entities(frames, states, roles, errors) / event_cue / score
- Emit JSONL files: 
 sections.jsonl:章节树
 paragraphs.jsonl：200-400 token段落块
 sentences.jsonl：句子级微块
"""

import argparse
import hashlib
import json
import os
import re
from typing import List, Dict, Optional, Tuple

from bs4 import BeautifulSoup, Tag
import requests
from tqdm import tqdm
import nltk
nltk.data.path.append('/root/nltk_data')

# ----- sentence splitter -----
try:
    from nltk.tokenize import sent_tokenize
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

def split_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    if _HAS_NLTK:
        try:
            sents = sent_tokenize(text)
            return [re.sub(r'\s+', ' ', s).strip() for s in sents if s.strip()]
        except Exception:
            pass
    # fallback: split on . ! ? ; : followed by space+capital or end
    return [p.strip() for p in re.split(r'(?<=[\.\!\?\;\:])\s+(?=[A-Z(])', text) if p.strip()]

def clean_text(t: str) -> str:
    t = t.replace('\u00ad', '')  # soft hyphen
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

###在构建 sentences.jsonl 时，为每句生成 eid = blake2s( section|anchor|para_idx|sent_idx|normalize(text) )
##normalize(text)：小写、去多空格、去软连字符/零宽字符。

def norm_text(t: str) -> str:
    t = t.replace("\u00ad","")  # soft hyphen
    t = re.sub(r"[\u200b-\u200d\uFEFF]", "", t)  # zero-width
    t = re.sub(r"\s+", " ", t.lower()).strip()
    return t

def make_eid(section, anchor, para_idx, sent_idx, text):
    key = f"{section}|{anchor or ''}|{para_idx}|{sent_idx}|{norm_text(text)}"
    return hashlib.blake2s(key.encode("utf-8"), digest_size=8).hexdigest()


# ----- seed (minimal) -----
#自动标注+人工复核
SEED = {
    "frames": [
        "STREAM","RESET_STREAM","STOP_SENDING","STREAM_DATA_BLOCKED",
        "DATA_BLOCKED","MAX_DATA","MAX_STREAM_DATA","MAX_STREAMS","STREAMS_BLOCKED"
    ],
    "states_sender": ["Ready","Send","Data Sent","Data Recvd","Reset Sent","Data Recvd","Reset Recvd"],
    "states_receiver": ["Recv","Size Known","Data Recvd","Data Read","Reset Recvd","Reset Read"],
    "roles": ["client","server","endpoint","peer","sender","receiver","initiator","responder","the sending part of the stream","the receiving part of the stream","The sender of a stream","The receiver of a stream"],
    "errors": ["FLOW_CONTROL_ERROR","FINAL_SIZE_ERROR"]
}

# SEED = {
#     "frames": [
#         "STREAM","RESET_STREAM","STOP_SENDING","STREAM_DATA_BLOCKED",
#         "DATA_BLOCKED","MAX_DATA","MAX_STREAM_DATA","MAX_STREAMS","STREAMS_BLOCKED"
#     ],
#     "states_sender": ["Ready","Send","Data Sent","Data Recvd","Reset Sent"],
#     "states_receiver": ["Recv","Size Known","Data Recvd","Data Read","Reset Recvd","Reset Read"],
#     "roles": ["client","server","endpoint","peer","sender","receiver","initiator","responder"],
#     "errors": ["FLOW_CONTROL_ERROR","FINAL_SIZE_ERROR"],
#     "flow_terms": ["credit","window","limit"],
#     "congestion_terms": ["cwnd","ssthresh","srtt","rttvar","PTO","loss","ECN","recovery"],
# }

RFC2119_REGEX = re.compile(
    r"\b(MUST(?:\s+NOT)?|SHOULD(?:\s+NOT)?|MAY|REQUIRED|RECOMMENDED|NOT\s+RECOMMENDED|OPTIONAL|SHALL(?:\s+NOT)?)\b"
)
EVENT_CUE_REGEX = re.compile(
    r"\b(upon|when|whenever|if|then|receive[sd]?|send[sd]?|ack(?:nowledg\w*)?|fin\b|timeout|expires?)\b",
    flags=re.IGNORECASE
)
ALLCAPS_TOKEN = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
STATE_NAME_RE = re.compile(
    r"\b(Ready|Send|Data Sent|Data Recvd|Reset Sent|Recv|Size Known|Data Read|Reset Recvd|Reset Read)\b"
)
ROLE_RE = re.compile(
    r"\b(client|server|endpoint|peer|sender|receiver|initiator|responder)s?\b",
    flags=re.IGNORECASE
)

# ----- facet heuristics -----
##同一个句子属于多个类别，这里先按优先级顺序选一个。低优先级，句子描述会变少。
###facet（主题面向）：基于所在章节路径 + 关键触发词判断：
# streams（§2.*；含“stream”大量出现且非帧定义）
# state_machine（§3.* 或含 “state/enter/leave/in the … state/transition”）
# flow_control（§4.* 或含 “credit/limit/MAX_DATA/blocked”）
# frames（§19.* 或句中含全大写帧名）
# terminology（§1.2 或含 “term/definition/defined as”）
# other（其他）
def guess_facet(section: str, text: str) -> str:
    sec_prefix = section.split('.')[0] if section else ""
    low = text.lower()
    if section.startswith("1.2") or "terminology" in low:
        return "terminology"
    if sec_prefix == "2":
        return "streams"
    if sec_prefix == "3":
        return "state_machine"
    if sec_prefix == "4" or "flow control" in low or "credit" in low:
        return "flow_control"
    if sec_prefix == "19" or "frame" in low:
        return "frames"
    if "state" in low or "transition" in low:
        return "state_machine"
    return "other"

def nearest_anchor(node: Tag) -> Optional[str]:

    if isinstance(node, Tag) and node.has_attr("id"):
        return f"#{node['id']}"
    a = node.find("a", attrs={"id": True}) or node.find("a", attrs={"name": True}) if isinstance(node, Tag) else None
    if a:
        return f"#{a.get('id') or a.get('name')}"
    # climb up to 3 levels
    parent = node
    for _ in range(3):
        if not isinstance(parent, Tag): break
        parent = parent.parent
        if isinstance(parent, Tag) and parent.has_attr("id"):
            return f"#{parent['id']}"
    return None

# ----- section parsing -----
SECTION_HDR = re.compile(r'^\s*(\d+(?:\.\d+)*)\s*[)\.]?\s+(.*)$')
#抽取章节：对每个h2/h3/h5形成节点；archor:页面中的id/name(#name-sending-stream-states)
def parse_sections(soup: BeautifulSoup) -> List[Dict]:
    """Build ordered section list by scanning h2..h5 that look like '3.1 Title'."""
    headers = soup.find_all(re.compile(r"^h[2-5]$"))
    sections = []
    order = 0
    for h in headers:
        txt = h.get_text(" ", strip=True)
        m = SECTION_HDR.match(txt)
        if not m:
            continue
        secnum, title = m.group(1), m.group(2).strip()
        order += 1
        sections.append({
            "section": secnum,
            "title": title,
            "anchor": nearest_anchor(h) or (f"#{h['id']}" if h.has_attr("id") else ""),
            "order": order, 
            "tag": h
        })
    return sections

def walk_blocks_with_section(soup: BeautifulSoup, sections: List[Dict]) -> List[Dict]:
    """
    Single pass DOM walk: maintain current_section by last seen header.
    Collect paragraph-like blocks with their owning section.
    (No reliance on sourceline.)

    单次遍历DOM：通过上一次看到的标题维护current_section。
    收集段落块及其所属节。
    """
    if soup.body:
        root = soup.body
    else:
        root = soup

    # Build a set of header tags we recognized
    header_tags = {rec["tag"] for rec in sections}
    section_by_tag = {rec["tag"]: rec for rec in sections}

    allowed = {"p","li","pre","figcaption","table","div"}
    blocks = []
    current = None

    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        # header?
        if el in header_tags:
            current = section_by_tag[el]
            continue
        # collect block text
        if el.name in allowed:
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt:
                continue
            if current is None:
                continue
            blocks.append({
                "section": current["section"],
                "anchor": current["anchor"],
                "title": current["title"],
                "tag_name": el.name,
                "text": txt
            })
    return blocks

# ----- entity tagging & scoring -----
#实体标记和打分
def auto_entities(text: str) -> Dict[str, List[str]]:
    caps = ALLCAPS_TOKEN.findall(text)
    frames = sorted(set([x for x in caps if x in SEED["frames"]]))   #正则捕获种子帧名（全大写词）
    errors = sorted(set([x for x in caps if x in SEED["errors"]]))   
    states = sorted(set(STATE_NAME_RE.findall(text)))  #匹配状态名/图题里的状态名
    roles = sorted(set([r.lower() for r in ROLE_RE.findall(text)])) #匹配角色名
    return {"frames": frames, "states": states, "roles": roles, "errors": errors}

def score_sentence(text: str, facet: str, normative: bool, entities: Dict[str, List[str]], event: bool) -> int:
    #score（证据密度分）：简单线性加权：
        #+2：含帧名或状态名
        # +2：normative==true
        # +1：含事件触发词
        # +1：位于段首/图题/定义句
    s = 0
    if entities.get("frames"): s += 2  # 
    if entities.get("states"): s += 2  
    if normative: s += 2
    if event: s += 1
    if facet in {"state_machine","frames","flow_control"}: s += 1
    return s

# ----- main pipeline -----
def run_pipeline(html: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    soup = BeautifulSoup(html, "lxml")

    # 1) sections
    sections = parse_sections(soup)
    with open(os.path.join(outdir, "sections.jsonl"), "w", encoding="utf-8") as f:
        for rec in sections:
            f.write(json.dumps({
                "section": rec["section"],
                "title": rec["title"],
                "anchor": rec["anchor"],
                "order": rec["order"]
            }, ensure_ascii=False) + "\n")

    # 2) paragraphs (200–400 token目标，这里先按自然块保留，后续可再重切)200–400 tokens，overlap 60–100
    raw_blocks = walk_blocks_with_section(soup, sections)
    paragraphs = []
    pid = 0
    for b in raw_blocks:
        pid += 1
        paragraphs.append({
            "para_id": f"{b['section']}-p{pid}",
            "section": b["section"],
            "anchor": b["anchor"],
            "text_block": b["text"],
            "tag": b["tag_name"]
        })
    with open(os.path.join(outdir, "paragraphs.jsonl"), "w", encoding="utf-8") as f:
        for rec in paragraphs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3) sentences + auto tags
    sentences = []
    sid_counter = 0
    for p in tqdm(paragraphs, desc="Splitting sentences"):
        sents = split_sentences(p["text_block"])
        for i, s in enumerate(sents, start=1):
            sid_counter += 1
            sid = f"{p['section']}-p{sid_counter}-s{i}"
            facet = guess_facet(p["section"], s)
            modalities = RFC2119_REGEX.findall(s)
            normative = bool(modalities)
            entities = auto_entities(s)
            event = bool(EVENT_CUE_REGEX.search(s))
            score = score_sentence(s, facet, normative, entities, event)
            eid = make_eid(p["section"], p["anchor"], sid_counter, i, s)
            #输出字段   
            sentences.append({
                "eid": eid,  #唯一标识符
                "sent_id": sid,
                "section": p["section"],
                "anchor": p["anchor"],
                "para_id": p["para_id"],
                "text": s,
                "facet": facet, #主题类别：基于所在章节路径和关键触发词判断   
                "normative": normative,  #是否含 RFC2119 词，出现就是true，并且记录modalities
                "modalities": modalities, #RFC2119 词列表
                "entities": entities, #自动识别的实体
                "event_cue": event, #是否含事件触发词
                "score": score #综合评分
            })

    with open(os.path.join(outdir, "sentences.jsonl"), "w", encoding="utf-8") as f:
        for rec in sentences:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 4) console summary
    print(f"[DONE] sections={len(sections)} paragraphs={len(paragraphs)} sentences={len(sentences)}")
    top = sorted(
        [s for s in sentences if s["normative"] and (s["entities"]["frames"] or s["facet"] in {"state_machine","frames"})],
        key=lambda x: (-x["score"])
    )[:8]
    for s in top:
        print(f"[{s['section']} {s['facet']} score={s['score']}] {s['text']}")
        print(f"  anchor={s['anchor']} modalities={s['modalities']} frames={s['entities']['frames']} states={s['entities']['states']}\n")

def main():
    html_path = "d:/ProtocolAutoModel-main/rfc_html/RFC_9000_ QUIC.html"  
    outdir = "d:/ProtocolAutoModel-main/data/rfc_9000_eid_corpus"          

    if html_path and os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    else:
        raise SystemExit("请确保 html_path 路径存在")
    
    run_pipeline(html, outdir)
    

    

if __name__ == "__main__":
    main()
