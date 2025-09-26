#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-generate a minimal QUIC seed lexicon from RFC9000 (transport) and RFC9002 (congestion control), Prefer HTML.

What it extracts (fully automatic, no manual labels):
- Frames (RFC9000 §19.*): ALLCAPS tokens, filtered to stream-related frames.
- Stream states (RFC9000 §3.1/§3.2): extracted from text patterns like "enter ... state", "in the ... state", etc.
- Roles & synonyms (RFC9000 §1.2 + §2.*): client/server/endpoint/peer/initiator/responder/sender/receiver with co-occurrence-based synonym hints.
- Error codes: ALLCAPS tokens ending with `_ERROR`, scored by co-occurrence with §4.* (Flow Control).
- Mechanism terms:
  * Flow control (RFC9000 §4.*): frequent MAX_* / *_BLOCKED frames + lowercase terms (credit, window, limit).
  * Congestion control (RFC9002): cwnd, ssthresh, srtt, rttvar, PTO, ECN, loss, recovery, detected automatically.

Output: a JSON file (seed_lexicon.json by default).
"""

import argparse
import collections
import json
import re
from typing import List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

# ----------------- Utilities -----------------

SENT_SPLIT = re.compile(r'(?<=[\.\?\!;:])\s+(?=[A-Z(])')

def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter; good enough for RFC prose."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    parts = SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]

def clean(t: str) -> str:
    t = t.replace('\u00ad', '')  # soft hyphen
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def fetch_html(path_or_url: str) -> BeautifulSoup:
    # 判断是否为本地文件路径
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return BeautifulSoup(r.text, 'lxml')
    else:
        with open(path_or_url, "r", encoding="utf-8", errors="ignore") as f:
            return BeautifulSoup(f.read(), 'lxml')
# ...existing code...

SEC_RE = re.compile(r'^\s*(\d+(?:\.\d+)*)\s*[)\.]?\s+(.*)$')

def parse_sections(soup: BeautifulSoup) -> List[Dict]:
    """Return ordered list of sections: {num,title,anchor,header_tag}"""
    out = []
    for h in soup.find_all(re.compile(r'^h[2-5]$')):
        txt = h.get_text(" ", strip=True)
        m = SEC_RE.match(txt)
        if not m:
            continue
        sec = m.group(1)
        title = m.group(2).strip()
        anchor = f"#{h.get('id')}" if h.has_attr('id') else None
        out.append({"num": sec, "title": title, "anchor": anchor, "tag": h})
    return out

def elements_until(next_stop: Optional[Tag], start: Tag):
    """Yield elements after 'start' until reaching next_stop header."""
    for el in start.next_elements:
        if next_stop and el == next_stop:
            break
        yield el

def collect_text_between(h: Tag, next_h: Optional[Tag]) -> List[str]:
    """Collect paragraph-like texts following header h, up to next header."""
    texts = []
    allowed = {"p", "li", "pre", "figcaption", "table", "div"}
    for el in elements_until(next_h, h):
        if isinstance(el, Tag) and el.name in allowed:
            txt = clean(el.get_text(" ", strip=True))
            if txt:
                texts.append(txt)
    return texts

# ----------------- Extraction heuristics -----------------

ALLCAPS_TOKEN = re.compile(r'\b([A-Z][A-Z0-9_]{2,})\b')
FLOW_CTRL_LOWER = {"credit", "window", "limit"}  # lowercase terms to consider in §4
STREAM_FRAME_WHITELIST = re.compile(
    r'^(STREAMS?_BLOCKED|STREAM_DATA_BLOCKED|DATA_BLOCKED|MAX_DATA|MAX_STREAM_DATA|MAX_STREAMS|STREAM|RESET_STREAM|STOP_SENDING)$'
)

STATE_CUE_SENT = re.compile(
    r'\b(?:enter(?:s|ed)?|transition(?:s|ed)?\s+to|leave(?:s|d)?|in|remain(?:s|ed)?\s+in)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|Recvd|Sent|Read|Known))?)\b'
)

STATE_CUE_STATE_WORD = re.compile(
    r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|Recvd|Sent|Read|Known))?)\s+state\b'
)

ROLES = ["client","server","endpoint","peer","initiator","responder","sender","receiver"]

def extract_frames_from_rfc9000(sections: List[Dict]) -> List[str]:
    frames = collections.Counter()
    for i, meta in enumerate(sections):
        sec = meta["num"]
        if not sec.startswith("19"):  # only §19.x frame definitions
            continue
        h = meta["tag"]
        next_h = sections[i+1]["tag"] if i+1 < len(sections) else None
        texts = collect_text_between(h, next_h)
        blob = " ".join(texts)
        for tok in ALLCAPS_TOKEN.findall(blob):
            if STREAM_FRAME_WHITELIST.match(tok):
                frames[tok] += 1
    # sort by frequency
    return [k for k, _ in frames.most_common()]

def extract_states_from_rfc9000(sections: List[Dict]) -> Tuple[List[str], List[str]]:
    """Return (sender_states, receiver_states) from §3.1 and §3.2."""
    sender = set()
    receiver = set()

    # gather text chunks for 3.1 and 3.2
    chunks = {"3.1": "", "3.2": ""}

    for i, meta in enumerate(sections):
        sec = meta["num"]
        if sec.startswith("3.1") or sec.startswith("3.2"):
            h = meta["tag"]
            next_h = sections[i+1]["tag"] if i+1 < len(sections) else None
            texts = collect_text_between(h, next_h)
            chunks["3.1" if sec.startswith("3.1") else "3.2"] += " " + " ".join(texts)

    def mine_states(text: str) -> List[str]:
        cands = set()
        for s in split_sentences(text):
            for m in STATE_CUE_SENT.finditer(s):
                cands.add(m.group(1))
            for m in STATE_CUE_STATE_WORD.finditer(s):
                cands.add(m.group(1))
        # Normalize spacing (e.g., "Data Sent", "Size Known")
        out = set()
        for c in cands:
            c = re.sub(r'\s+', ' ', c).strip()
            # keep only plausible state names (short, with certain roots)
            if any(root in c for root in ["Ready","Send","Data","Recv","Read","Known","Reset","Size"]):
                if 3 <= len(c) <= 20 and len(c.split()) <= 2:
                    out.add(c)
        return sorted(out, key=lambda x: (len(x), x))

    sender_states = mine_states(chunks["3.1"])
    receiver_states = mine_states(chunks["3.2"])

    # Heuristic: classify Data Sent/Recvd/Read, Ready/Send into sender/receiver buckets
    # Keep mined lists, but also split by which section they came from.
    sender.update(sender_states)
    receiver.update(receiver_states)

    # Return as lists with stable order
    return sorted(sender), sorted(receiver)

def extract_roles_synonyms(sections: List[Dict]) -> Dict[str, List[str]]:
    """
    Scan §1.2 and §2.* for roles and try to propose synonyms via co-occurrence:
    - "initiator" ↔ "client", "responder" ↔ "server", "endpoint" ↔ "peer"
    """
    texts = []
    for i, meta in enumerate(sections):
        sec = meta["num"]
        if sec.startswith("1.2") or sec.startswith("2"):
            h = meta["tag"]
            next_h = sections[i+1]["tag"] if i+1 < len(sections) else None
            texts.extend(collect_text_between(h, next_h))

    pairs = collections.Counter()
    for para in texts:
        sent_list = split_sentences(para)
        for s in sent_list:
            low = s.lower()
            present = [r for r in ROLES if r in low]
            # count pairs co-occurring within a sentence
            for i in range(len(present)):
                for j in range(i+1, len(present)):
                    a, b = sorted([present[i], present[j]])
                    pairs[(a, b)] += 1

    # propose synonyms based on strongest pairs we expect
    syns = {}
    def maybe(a, b):
        if pairs.get((a, b), 0) > 0:
            syns[b] = a

    maybe("client", "initiator")
    maybe("server", "responder")
    maybe("peer", "endpoint")  # endpoint≈peer（选“peer”为规范名时）
    # sender/receiver 作为“局部角色”，通常不并入 client/server

    return syns

def extract_errors(sections: List[Dict]) -> List[str]:
    """
    ALLCAPS + _ERROR tokens, scored by co-occurrence in §4.* (Flow Control).
    """
    all_errors = collections.Counter()
    flow_errors = collections.Counter()

    for i, meta in enumerate(sections):
        sec = meta["num"]
        h = meta["tag"]
        next_h = sections[i+1]["tag"] if i+1 < len(sections) else None
        texts = collect_text_between(h, next_h)
        blob = " ".join(texts)
        for tok in ALLCAPS_TOKEN.findall(blob):
            if tok.endswith("_ERROR"):
                all_errors[tok] += 1
                if sec.startswith("4"):
                    flow_errors[tok] += 1

    # score: prioritize those seen in §4.* (flow control)
    scored = []
    for tok, c in all_errors.items():
        score = 5 * flow_errors.get(tok, 0) + c
        scored.append((score, tok))
    scored.sort(reverse=True)
    return [t for _, t in scored[:4]]  # keep top few; you can reduce to 2 if需要更小

def extract_flow_mechanism_terms(sections: List[Dict]) -> Dict[str, List[str]]:
    """
    From RFC9000 §4.*:
      - MAX_* / *_BLOCKED (ALLCAPS)
      - lowercase terms credit/window/limit
    """
    caps = collections.Counter()
    lowers = collections.Counter()

    for i, meta in enumerate(sections):
        sec = meta["num"]
        if not sec.startswith("4"):
            continue
        h = meta["tag"]
        next_h = sections[i+1]["tag"] if i+1 < len(sections) else None
        texts = collect_text_between(h, next_h)
        blob = " ".join(texts)
        for tok in ALLCAPS_TOKEN.findall(blob):
            if tok.startswith("MAX_") or tok.endswith("_BLOCKED"):
                caps[tok] += 1
        low = blob.lower()
        for w in FLOW_CTRL_LOWER:
            if w in low:
                lowers[w] += low.count(w)

    return {
        "caps": [k for k, _ in caps.most_common()],
        "lowers": [k for k, _ in lowers.most_common()]
    }

def extract_congestion_terms_9002(soup_9002: BeautifulSoup) -> List[str]:
    """
    Scan RFC9002 for presence of common congestion control terms.
    We don't hardcode outputs—just detect presence and keep encountered ones.
    """
    targets = ["cwnd", "ssthresh", "srtt", "rttvar", "PTO", "ECN", "loss", "recovery"]
    text = clean(soup_9002.get_text(" ", strip=True))
    found = []
    for t in targets:
        pat = re.compile(rf'\b{re.escape(t)}\b')
        if pat.search(text):
            found.append(t)
    # prefer a stable informative order
    order = ["cwnd","ssthresh","srtt","rttvar","PTO","ECN","loss","recovery"]
    found_sorted = [t for t in order if t in found]
    return found_sorted

# ----------------- Main -----------------



def main():
    # 直接指定本地文件路径和输出路径
    rfc9000_path = "d:/ProtocolAutoModel-main/rfc_html/RFC_9000_ QUIC.html"
    rfc9002_path = "d:/ProtocolAutoModel-main/rfc_html/RFC_9002_ QUIC.html"
    out_path = "d:/ProtocolAutoModel-main/data/minimal_corpus/seed_lexicon.json"

    print("[*] Fetching RFC9000 (HTML)…")
    soup9000 = fetch_html(rfc9000_path)
    sec9000 = parse_sections(soup9000)
    print(f"    Sections parsed: {len(sec9000)}")

    print("[*] Extracting frames from §19.* …")
    frames = extract_frames_from_rfc9000(sec9000)

    print("[*] Mining stream states from §3.1/§3.2 …")
    sender_states, receiver_states = extract_states_from_rfc9000(sec9000)

    print("[*] Mining roles & synonyms from §1.2 and §2.* …")
    role_synonyms = extract_roles_synonyms(sec9000)

    print("[*] Mining error codes (with flow control co-occurrence scoring) …")
    errors = extract_errors(sec9000)

    print("[*] Mining flow-control mechanism terms from §4.* …")
    flow_terms = extract_flow_mechanism_terms(sec9000)

    print("[*] Fetching RFC9002 (HTML) for congestion terms …")
    soup9002 = fetch_html(rfc9002_path)
    congestion_terms = extract_congestion_terms_9002(soup9002)

    seed = {
        "protocol": "QUIC",
        "docs": {"rfc9000": rfc9000_path, "rfc9002": rfc9002_path},
        "frames": frames,
        "states": {"sender": sender_states, "receiver": receiver_states},
        "roles": {
            "canonical": ["client","server","endpoint","peer","sender","receiver"],
            "synonyms": role_synonyms
        },
        "errors": errors,
        "mechanisms": {
            "flow_control_caps": flow_terms["caps"],
            "flow_control_terms": flow_terms["lowers"],
            "congestion_terms": congestion_terms
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Seed lexicon written to {out_path}")
    print("     frames           :", frames[:12])
    print("     sender states    :", sender_states)
    print("     receiver states  :", receiver_states)
    print("     role synonyms    :", role_synonyms)
    print("     errors           :", errors)
    print("     flow caps        :", flow_terms['caps'][:12])
    print("     flow lowers      :", flow_terms['lowers'])
    print("     congestion terms :", congestion_terms)

if __name__ == "__main__":
    main()
