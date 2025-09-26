# pip install sentence-transformers faiss-cpu whoosh

from sentence_transformers import SentenceTransformer
import faiss, json, os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD
from whoosh.qparser import MultifieldParser

# ---------- 1) 读数据 ----------
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

sent_items = list(load_jsonl("d:/ProtocolAutoModel-main/data/sentences.jsonl"))   # 每条有 eid/text/section/... 等
para_items = list(load_jsonl("d:/ProtocolAutoModel-main/data/paragraphs.jsonl"))  # para_id/text_block/section/...

# ---------- 2) 构造嵌入文本 ----------
def make_embed_text_sent(x):
    facets = ",".join([k for k,v in x.get("facets",{}).items() if v])
    ents = []
    for k,v in x.get("entities",{}).items():
        ents += v if isinstance(v, list) else [v]
    ents = ",".join(sorted(set(ents)))
    title = f"[SECTION {x.get('section','?')}]"
    return f"{title}\nSentence: {x['text']}\nFacets:{facets}\nEntities:{ents}"

def make_embed_text_para(p):
    return f"[SECTION {p.get('section','?')}]\nParagraph: {p['text_block']}"

# ---------- 3) 生成向量 ----------
model = SentenceTransformer("intfloat/e5-base-v2")  # 可替换
def emb(texts, bsz=64):
    vecs = model.encode([f"passage: {t}" for t in texts], batch_size=bsz, normalize_embeddings=True)
    return vecs.astype('float32')

sent_texts = [make_embed_text_sent(x) for x in sent_items]
para_texts = [make_embed_text_para(p) for p in para_items]

sent_vecs = emb(sent_texts)
para_vecs = emb(para_texts)

# ---------- 4) 建 FAISS ----------
d = sent_vecs.shape[1]
faiss_sent = faiss.IndexFlatIP(d)
faiss_para = faiss.IndexFlatIP(d)
faiss_sent.add(sent_vecs)
faiss_para.add(para_vecs)

# 保存映射
id2meta_sent = sent_items     # index i -> metadata (含 eid)
id2meta_para = para_items

# ---------- 5) 建 Whoosh 倒排 ----------
schema = Schema(
    id=ID(stored=True, unique=True),
    section=TEXT(stored=True),
    text=TEXT(stored=True),
    tokens=KEYWORD(stored=True, commas=True)  # ALLCAPS/帧名等
)
os.makedirs("lex_index", exist_ok=True)
ix = create_in("lex_index", schema)
writer = ix.writer()
def allcaps_tokens(s):
    import re
    return ",".join(sorted(set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", s))))
for i, x in enumerate(sent_items):
    toks = allcaps_tokens(x["text"])
    writer.add_document(id=str(i), section=x.get("section",""), text=x["text"], tokens=toks)
writer.commit()

# ---------- 6) 检索（融合示例） ----------
def search(query_text, section_hint=None, topk=10, alpha=0.7):
    # 先倒排筛（可选）
    qp = MultifieldParser(["text","tokens","section"], schema=ix.schema)
    q = qp.parse(query_text + (f" section:{section_hint}" if section_hint else ""))
    with ix.searcher() as s:
        res = s.search(q, limit=200)
        cand_ids = [int(r["id"]) for r in res]

    # 若无筛选结果则全量向量检索
    qv = emb([query_text])[0][None, :]
    if cand_ids:
        cand_vecs = sent_vecs[cand_ids]
        sims = (qv @ cand_vecs.T)[0]
        order = sims.argsort()[::-1][:topk]
        hits = [(cand_ids[i], float(sims[i])) for i in order]
    else:
        sims, idxs = faiss_sent.search(qv, topk)
        hits = [(int(idxs[0][i]), float(sims[0][i])) for i in range(topk)]
    # 返回元数据（含 eid）
    return [id2meta_sent[i] for i,_ in hits]

# ---------- 7) 盲补用法 ----------
# 例：to_state 缺失 → 构造查询
q = 'section:3 sender "enter" STREAM FIN MUST'
blind_evidence = search(q, section_hint="3")