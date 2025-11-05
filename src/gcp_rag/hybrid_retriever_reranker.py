# hybrid_retriever_reranker.py
# pip install --upgrade google-cloud-aiplatform
# gcloud auth application-default login
# export PROJECT_ID=your-project ; export LOCATION=us-central1

import os
import re
import math
import time
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Part

# ----------------------------
# Config
# ----------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "YOUR_PROJECT")
LOCATION   = os.getenv("LOCATION",   "us-central1")
CORPUS     = os.getenv("RAG_CORPUS", "uhc-coverage-change-corpus")

# Retrieval knobs
TOP_K_DENSE = 20             # initial dense recall from RAG
KEYWORD_HARD_KEEP = 0.25     # keep a candidate if keyword score >= this
MAX_PER_DOC = 3              # after dedup, how many chunks per doc/URI to keep
MMR_K = 10                   # select K after MMR
MMR_LAMBDA = 0.7             # 1.0=all similarity, 0.0=all diversity

# Rerank knobs
LLM_MODEL = os.getenv("RERANK_MODEL", "gemini-1.5-pro")
RERANK_TOP = 5               # how many after rerank to keep
BOOST_SECTIONS = {"coverage rationale","applicable codes","benefit considerations"}

# ----------------------------
# Helpers
# ----------------------------

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def cosine_to_sim(score: float) -> float:
    # RAG returns "score" (higher is better). We'll min-max later; treat as similarity proxy.
    return float(score or 0.0)

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def tokenize_for_overlap(text: str) -> set:
    terms = re.findall(r"[A-Za-z0-9\-]+", (text or "").lower())
    return set(t for t in terms if len(t) > 2)

# ----------------------------
# Keyword extraction (domain-aware but tiny)
# ----------------------------

ALIASES = {
    "prior auth": {"pa", "authorization", "prior authorization"},
    "hospital outpatient": {"hopd", "hospital outpatient"},
    "site of service": {"site of service", "sos"},
    "rituximab": {"rituximab", "rituxan", "riabni", "ruxience", "truxima"},
    "mri": {"mri"},
    "ct": {"ct", "computed tomography"},
    "codes": {"applicable codes", "cpt", "hcpcs", "icd"},
    "preferred": {"preferred", "non-preferred"},
}

BIGRAMS = [
    "prior auth", "site of service", "coverage rationale",
    "applicable codes", "benefit considerations",
]

STOP = {
    "what","is","are","the","for","a","an","and","or","to","of","in","as","on","by",
    "with","from","does","do","did","be","can","i","we","they","you","me"
}

def build_keywords(query: str) -> List[str]:
    q = normalize(query).lower()
    # harvest bigrams first
    kws = set()
    for bg in BIGRAMS:
        if bg in q:
            kws.add(bg)
    # unigrams
    for tok in re.findall(r"[a-z0-9\-]+", q):
        if len(tok) <= 2 or tok in STOP:
            continue
        kws.add(tok)
        # alias expand
        for key, vals in ALIASES.items():
            if tok in vals or tok == key:
                kws |= vals
                kws.add(key)
    return sorted(kws)

# ----------------------------
# RAG Managed DB: dense retrieval
# ----------------------------

@dataclass
class Hit:
    text: str
    score: float
    uri: str
    # optional metadata if present (RAG citations sometimes include attributes)
    section: Optional[str] = None
    page: Optional[int] = None
    version_date: Optional[str] = None
    effective_from: Optional[str] = None
    policy_id: Optional[str] = None
    chunk_id: Optional[str] = None

def dense_retrieve(query: str, top_k: int = TOP_K_DENSE, corpus: str = CORPUS) -> List[Hit]:
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    cfg = rag.RagRetrievalConfig(top_k=top_k)
    resp = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus)],
        text=query,
        rag_retrieval_config=cfg,
    )
    out: List[Hit] = []
    for c in (resp.context.citations or []):
        out.append(Hit(
            text = c.content or "",
            score = c.score or 0.0,
            uri = c.uri or "",
            # If you later switch to Vector Search or attach metadata, map it here:
            # section = c.metadata.get("section") if c.metadata else None,
        ))
    return out

# ----------------------------
# Keyword post-filter + de-dup + MMR diversity
# ----------------------------

def keyword_score(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    t = (text or "").lower()
    hits = 0
    for kw in keywords:
        if kw in t:
            hits += 1
    return hits / max(1, len(keywords))

def dedup_by_doc(hits: List[Hit], max_per_doc: int = MAX_PER_DOC) -> List[Hit]:
    buckets: Dict[str, List[Hit]] = {}
    for h in hits:
        buckets.setdefault(h.uri, []).append(h)
    kept: List[Hit] = []
    for uri, hs in buckets.items():
        hs_sorted = sorted(hs, key=lambda x: x.score, reverse=True)
        kept.extend(hs_sorted[:max_per_doc])
    return kept

def mmr_select(hits: List[Hit], k: int = MMR_K, lam: float = MMR_LAMBDA) -> List[Hit]:
    """
    Maximal Marginal Relevance using text Jaccard for diversity + score for similarity.
    """
    if not hits:
        return []
    # Normalize similarity (score) 0..1
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    sims = [ (s - lo) / (hi - lo + 1e-9) for s in scores ]

    texts = [tokenize_for_overlap(h.text) for h in hits]

    selected = []
    candidate_idx = list(range(len(hits)))

    # pick best by similarity first
    first = max(range(len(hits)), key=lambda i: sims[i])
    selected.append(first)
    candidate_idx.remove(first)

    while len(selected) < min(k, len(hits)) and candidate_idx:
        best_i, best_score = None, -1.0
        for i in candidate_idx:
            # diversity = max Jaccard similarity against already selected (we penalize this)
            div = 0.0
            for j in selected:
                div = max(div, jaccard(texts[i], texts[j]))
            mmr = lam * sims[i] - (1 - lam) * div
            if mmr > best_score:
                best_score, best_i = mmr, i
        selected.append(best_i)
        candidate_idx.remove(best_i)

    return [hits[i] for i in selected]

def hybrid_retrieve(query: str) -> List[Hit]:
    kws = build_keywords(query)
    dense = dense_retrieve(query, top_k=TOP_K_DENSE, corpus=CORPUS)

    # keyword keep/boost
    rescored: List[Tuple[float, Hit]] = []
    for h in dense:
        kscore = keyword_score(h.text, kws)
        sim = cosine_to_sim(h.score)
        # soft combine: boost sim by keyword presence
        comb = 0.8 * sim + 0.2 * kscore
        # but also apply a hard keep if strong keyword match
        if kscore >= KEYWORD_HARD_KEEP:
            comb = max(comb, sim + 0.1)
        rescored.append((comb, h))

    rescored.sort(key=lambda x: x[0], reverse=True)
    hits_sorted = [h for _, h in rescored]

    # dedup & diversify
    hits_dedup = dedup_by_doc(hits_sorted, max_per_doc=MAX_PER_DOC)
    hits_mmr  = mmr_select(hits_dedup, k=MMR_K, lam=MMR_LAMBDA)
    return hits_mmr

# ----------------------------
# Reranker (LLM answerability 0..5 + heuristic boosts)
# ----------------------------

RERANK_PROMPT = """You are a healthcare policy QA rater.
Score from 0 to 5 how well the CANDIDATE directly answers the QUESTION.
Guidelines:
0 = unrelated or no answerable content
3 = partially related or generic mention
5 = directly answerable (explicit criteria, coverage rules, preferred products, codes)

Return ONLY the integer 0..5.

QUESTION:
{q}

CANDIDATE:
{text}
"""

def llm_score_answerability(query: str, text: str, model_name: str = LLM_MODEL) -> float:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(model_name)
    prompt = RERANK_PROMPT.format(q=query, text=text[:6000])  # keep within prompt limits
    out = model.generate_content([Part.from_text(prompt)], generation_config={"temperature": 0})
    raw = (out.text or "").strip()
    m = re.search(r"\b([0-5])\b", raw)
    return float(m.group(1)) if m else 0.0

def heuristic_boost(text: str) -> float:
    t = (text or "").lower()
    # light boosts for canonical sections
    for s in BOOST_SECTIONS:
        if s in t:
            return 0.15
    # small boost if codes/jargon present
    if re.search(r"\b(cpt|hcpcs|icd|pos)\b", t):
        return 0.1
    return 0.0

def rerank_chunks(query: str, hits: List[Hit], top_n: int = RERANK_TOP) -> List[Dict[str, Any]]:
    if not hits:
        return []
    # normalize initial sim scores (0..1)
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    sims = [ (s - lo) / (hi - lo + 1e-9) for s in scores ]

    ranked = []
    for i, h in enumerate(hits):
        # LLM score is the most important
        a = llm_score_answerability(query, h.text)
        a_norm = a / 5.0
        boost = heuristic_boost(h.text)
        # combine: 0.7*LLM + 0.25*sim + 0.05*boost
        final = 0.70 * a_norm + 0.25 * sims[i] + 0.05 * boost
        ranked.append((final, h, a))
    ranked.sort(key=lambda x: x[0], reverse=True)
    out = []
    for final, h, a in ranked[:top_n]:
        out.append({
            "text": h.text,
            "uri": h.uri,
            "score_dense": h.score,
            "score_llm": a,
            "score_final": final,
            # If you later enrich metadata, add it here:
            "policy_id": h.policy_id,
            "version_date": h.version_date,
            "effective_from": h.effective_from,
            "section": h.section,
            "page": h.page,
            "chunk_id": h.chunk_id,
        })
    return out

# ----------------------------
# Demo runner
# ----------------------------

def demo():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    q = "As of Oct 1, 2025, what are the preferred rituximab products for non-oncology?"
    print("QUERY:", q)
    cand = hybrid_retrieve(q)
    print(f"Hybrid retrieved: {len(cand)} candidates")
    top = rerank_chunks(q, cand, top_n=5)
    print("\nTOP RESULTS:")
    for i, t in enumerate(top, 1):
        print(f"[{i}] final={t['score_final']:.3f}  dense={t['score_dense']:.3f}  llm={t['score_llm']:.1f}")
        print(f"uri={t['uri']}")
        print((t["text"] or "")[:400].strip(), "\n")

if __name__ == "__main__":
    if PROJECT_ID == "YOUR_PROJECT":
        raise SystemExit("Set PROJECT_ID/LOCATION/RAG_CORPUS env vars first.")
    demo()
