"""
UHC Change Intelligence CLI
============================
Usage:
    python app/app.py query "Does rituximab require prior auth?"
    python app/app.py query "What changed for bariatric surgery between Jan and Oct 2025?" --versions 1-1-2025 10-1-2025
    python app/app.py diff --policy UHC-PA-COMM --section "Bariatric surgery" --old 1-1-2025 --new 10-1-2025
    python app/app.py eval

Modes:
    --mode local   Use local JSONL index (default; no GCP credentials needed)
    --mode gcp     Use Vertex AI RAG + Gemini (requires GCP setup)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── path setup so we can import src/ without installing ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

CHUNKS_DIR     = ROOT / "data" / "chunks"
PARSED_DIR     = ROOT / "data" / "parsed"
SCHEMA_FILE    = ROOT / "schemas" / "response.schema.json"
EMBED_CACHE    = ROOT / "data" / "embeddings.npy"
EMBED_META     = ROOT / "data" / "embeddings_meta.json"

# Hybrid weighting: dense score weight vs BM25 score weight
DENSE_WEIGHT   = 0.6
BM25_WEIGHT    = 0.4
# MMR: balance relevance vs diversity (1.0 = pure relevance, 0.0 = pure diversity)
MMR_LAMBDA     = 0.7
EMBED_MODEL    = "all-MiniLM-L6-v2"   # 22 MB, fast, good semantic quality

# ─────────────────────────────────────────────────────────────────────────────
# Chunk loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """Load all chunks from JSONL files into memory."""
    docs: List[Dict[str, Any]] = []
    for p in sorted(chunks_dir.glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Dense retrieval — sentence-transformers + cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedder():
    """
    Lazy-load the sentence-transformers model (cached after first call).
    Returns None and falls back to BM25-only if the model cannot be loaded
    (not installed, no internet, or restricted network).
    """
    if not hasattr(_get_embedder, "_model"):
        try:
            from sentence_transformers import SentenceTransformer
            _get_embedder._model = SentenceTransformer(EMBED_MODEL)
        except Exception:
            _get_embedder._model = None
    return _get_embedder._model


def _load_or_build_embeddings(chunks: List[Dict[str, Any]]):
    """
    Return (matrix, chunk_ids) where matrix is (N, D) float32 numpy array.
    Builds and caches to disk on first call; reuses cache if chunk count matches.
    """
    import numpy as np

    model = _get_embedder()
    if model is None:
        return None, []

    # Cache hit: same number of chunks → reuse
    if EMBED_CACHE.exists() and EMBED_META.exists():
        meta = json.loads(EMBED_META.read_text())
        if meta.get("n_chunks") == len(chunks):
            matrix = np.load(str(EMBED_CACHE))
            return matrix, meta["chunk_ids"]

    # Build embeddings
    print("  [dense] Building embeddings for corpus … ", end="", flush=True)
    texts = [c.get("text", "") for c in chunks]
    matrix = model.encode(texts, batch_size=64, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    chunk_ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]

    np.save(str(EMBED_CACHE), matrix.astype("float32"))
    EMBED_META.write_text(json.dumps({"n_chunks": len(chunks), "chunk_ids": chunk_ids}))
    print(f"done ({len(chunks)} chunks, dim={matrix.shape[1]})")
    return matrix.astype("float32"), chunk_ids


def _dense_scores(query: str, matrix) -> "np.ndarray":
    """Return cosine similarities for query against all rows of matrix (already L2-normed)."""
    import numpy as np
    model = _get_embedder()
    if model is None or matrix is None:
        return np.zeros(0)
    q_vec = model.encode([query], normalize_embeddings=True)[0]   # shape (D,)
    return matrix @ q_vec   # cosine sim, shape (N,)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\-]+", (text or "").lower())


def _bm25_scores(query_tokens: List[str], doc_tokens_list: List[List[str]],
                 k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Vectorised BM25 over all documents, returns a score per doc."""
    import math
    N = len(doc_tokens_list)
    avgdl = sum(len(d) for d in doc_tokens_list) / max(1, N)

    # IDF per query term (proper IDF over corpus)
    df: Dict[str, int] = {}
    for tokens in doc_tokens_list:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    scores = [0.0] * N
    for t in set(query_tokens):
        idf = math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
        for i, tokens in enumerate(doc_tokens_list):
            tf = tokens.count(t)
            if tf == 0:
                continue
            dl = len(tokens)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avgdl)))
            scores[i] += idf * tf_norm
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# MMR diversity re-ranking
# ─────────────────────────────────────────────────────────────────────────────

def _mmr(query_vec, candidate_indices: List[int], matrix,
         k: int, lmbda: float = MMR_LAMBDA) -> List[int]:
    """
    Maximal Marginal Relevance: balance relevance to query vs diversity among
    selected chunks.  candidate_indices are pre-sorted by hybrid score.
    Returns up to k indices in MMR order.
    """
    import numpy as np
    if matrix is None or len(candidate_indices) == 0:
        return candidate_indices[:k]

    selected: List[int] = []
    remaining = list(candidate_indices)

    q_sims = (matrix[remaining] @ query_vec)  # cosine to query

    while remaining and len(selected) < k:
        if not selected:
            # First pick: highest query similarity
            best = int(np.argmax(q_sims))
        else:
            sel_mat = matrix[selected]
            # For each candidate: max cosine sim to already-selected
            max_sim_to_sel = (matrix[remaining] @ sel_mat.T).max(axis=1)
            mmr_scores = lmbda * q_sims - (1 - lmbda) * max_sim_to_sel
            best = int(np.argmax(mmr_scores))

        selected.append(remaining[best])
        remaining.pop(best)
        q_sims = np.delete(q_sims, best)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retriever: dense + BM25 → fuse → MMR
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache so embeddings are only built once per process
_embed_matrix = None
_embed_chunk_ids: List[str] = []


def local_retrieve(query: str, chunks: List[Dict[str, Any]],
                   k: int = 8) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
      1. Dense cosine similarity (sentence-transformers all-MiniLM-L6-v2)
      2. BM25 keyword matching with proper IDF
      3. Reciprocal Rank Fusion to combine scores
      4. MMR diversity re-ranking on the top-40 candidates
    Falls back to BM25-only if sentence-transformers is not installed.
    """
    import numpy as np

    global _embed_matrix, _embed_chunk_ids

    q_tokens = _tokenize(query)
    doc_tokens_list = [_tokenize(c.get("text", "")) for c in chunks]

    # ── BM25 ──────────────────────────────────────────────────────────────
    bm25_raw = _bm25_scores(q_tokens, doc_tokens_list)
    bm25_max = max(bm25_raw) if bm25_raw else 1.0
    bm25_norm = [s / max(bm25_max, 1e-9) for s in bm25_raw]

    # ── Dense ─────────────────────────────────────────────────────────────
    if _embed_matrix is None:
        _embed_matrix, _embed_chunk_ids = _load_or_build_embeddings(chunks)

    model = _get_embedder()
    use_dense = model is not None and _embed_matrix is not None and len(_embed_matrix) == len(chunks)

    if use_dense:
        q_vec = model.encode([query], normalize_embeddings=True)[0]
        dense_raw = (_embed_matrix @ q_vec).tolist()
        dense_min = min(dense_raw)
        dense_max = max(dense_raw)
        dense_norm = [(s - dense_min) / max(dense_max - dense_min, 1e-9) for s in dense_raw]
    else:
        dense_norm = [0.0] * len(chunks)
        q_vec = None

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────
    # RRF score = sum(1 / (rank + 60)) over each ranking
    def rrf_ranks(scores: List[float]) -> List[float]:
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rrf = [0.0] * len(scores)
        for rank, idx in enumerate(ranked):
            rrf[idx] = 1.0 / (rank + 60)
        return rrf

    rrf_dense = rrf_ranks(dense_norm)
    rrf_bm25  = rrf_ranks(bm25_norm)

    hybrid = [
        DENSE_WEIGHT * rrf_dense[i] + BM25_WEIGHT * rrf_bm25[i]
        for i in range(len(chunks))
    ]

    # Top-40 candidates for MMR
    top40 = sorted(range(len(chunks)), key=lambda i: hybrid[i], reverse=True)[:40]

    # ── MMR diversity ─────────────────────────────────────────────────────
    if use_dense and q_vec is not None:
        selected_indices = _mmr(q_vec, top40, _embed_matrix, k=k * 2)
    else:
        selected_indices = top40[: k * 2]

    # ── Build result list, limit per policy+version ───────────────────────
    seen: Dict[str, int] = {}
    results = []
    for idx in selected_indices:
        chunk = chunks[idx]
        key = f"{chunk.get('policy_id')}_{chunk.get('version_date')}"
        if seen.get(key, 0) >= 2:
            continue
        seen[key] = seen.get(key, 0) + 1
        result = dict(chunk)
        result["score"]       = round(hybrid[idx], 6)
        result["score_dense"] = round(dense_norm[idx] if use_dense else 0.0, 4)
        result["score_bm25"]  = round(bm25_norm[idx], 4)
        results.append(result)
        if len(results) >= k:
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Gemini generation (local mode — google-generativeai SDK, just needs API key)
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_LOCAL_MODEL  = "gemini-2.0-flash"
GEMINI_REST_URL     = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

_SYSTEM_PROMPT = """\
You are a healthcare coverage expert. Answer ONLY from the provided UHC policy excerpts.

Return a single JSON object — no markdown fences, no extra text — with these exact keys:
{
  "decision": "<Covered|RequiresPA|NotCovered|Ambiguous>",
  "validity_window": {"effective_from": "<YYYY-MM-DD or version string>", "effective_to": null},
  "evidence": [
    {"policy_id": "...", "section": "...", "page": <int>, "quote": "<verbatim excerpt>"}
  ],
  "changes": [
    {"section": "...", "old": "<prior wording>", "new": "<new wording>"}
  ],
  "notes": ["<any caveats>"]
}

Rules:
- Set decision=Ambiguous if the excerpts do not clearly answer the question.
- evidence must cite the policy_id, section, and page from the excerpts provided.
- changes should only be populated if the question asks about version differences.
- Do NOT hallucinate — every claim must trace to a provided excerpt.
"""


def _build_context(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for h in hits:
        header = (
            f"[{h.get('policy_id','')} | v{h.get('version_date','')} "
            f"| §{h.get('section','')} | p.{h.get('page','')}]"
        )
        parts.append(f"{header}\n{h.get('text','')}")
    return "\n\n---\n\n".join(parts)


def _gemini_generate(question: str, hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Call Gemini REST API directly (no SDK — just urllib + json).
    Returns the parsed response dict, or None if key is missing / call fails.
    """
    import urllib.request
    import urllib.error

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    user_prompt = (
        f"Policy excerpts:\n\n{_build_context(hits)}"
        f"\n\n---\nQuestion: {question}\n\nJSON answer:"
    )
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }).encode("utf-8")

    url = GEMINI_REST_URL.format(model=GEMINI_LOCAL_MODEL, key=api_key)
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        raw = body["candidates"][0]["content"]["parts"][0]["text"]
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        return json.loads(raw)
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback (used when no API key / generation fails)
# ─────────────────────────────────────────────────────────────────────────────

_PA_KEYWORDS  = re.compile(r"\bprior auth(?:orization)?\b|\bPA\b|\brequires? auth", re.I)
_NOT_COVERED  = re.compile(r"\bnot covered\b|\bno benefit\b|\bexcluded?\b", re.I)
_COVERED      = re.compile(r"\bcovered\b|\bno prior auth\b|\bno PA required\b", re.I)


def _infer_decision(query: str, hits: List[Dict[str, Any]]) -> str:
    combined = query + " " + " ".join(h.get("text", "") for h in hits[:3])
    if _PA_KEYWORDS.search(combined):
        return "RequiresPA"
    if _NOT_COVERED.search(combined):
        return "NotCovered"
    if _COVERED.search(combined):
        return "Covered"
    return "Ambiguous"


def _build_evidence(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ev = []
    for h in hits[:5]:
        quote = (h.get("text") or "")[:300].replace("\n", " ").strip()
        ev.append({
            "policy_id": h.get("policy_id", "unknown"),
            "section":   h.get("section", "unknown"),
            "page":      h.get("page") or 0,
            "quote":     quote,
            "version_date": h.get("version_date"),
            "score":     h.get("score"),
        })
    return ev


def _build_validity(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    dates = [h.get("effective_from") for h in hits if h.get("effective_from")]
    version_dates = [h.get("version_date") for h in hits if h.get("version_date")]
    effective = dates[0] if dates else (version_dates[0] if version_dates else "unknown")
    return {"effective_from": effective, "effective_to": None}


# ─────────────────────────────────────────────────────────────────────────────
# Diff helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_diff(policy_id: str, old_version: str, new_version: str,
                 section: Optional[str] = None) -> List[Dict[str, str]]:
    import difflib

    def load(version: str) -> Dict[str, str]:
        path = PARSED_DIR / f"{policy_id}_{version}.jsonl"
        sections: Dict[str, List[str]] = {}
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                title = (rec.get("section") or "").strip()
                sections.setdefault(title, []).append(rec.get("text", ""))
        return {k: "\n".join(v) for k, v in sections.items()}

    old_map = load(old_version)
    new_map = load(new_version)

    if not old_map and not new_map:
        return [{"error": f"No parsed data found for {policy_id} versions {old_version} / {new_version}"}]

    target_sections = [section] if section else list(set(old_map) | set(new_map))
    diffs = []
    for sec in sorted(target_sections):
        o = (old_map.get(sec) or "").splitlines()
        n = (new_map.get(sec) or "").splitlines()
        if o == n:
            continue
        diff_lines = list(difflib.unified_diff(
            o, n, lineterm="",
            fromfile=f"{old_version}/{sec}",
            tofile=f"{new_version}/{sec}",
            n=2
        ))
        if diff_lines:
            diffs.append({
                "section": sec,
                "old": "\n".join(o[:10]),
                "new": "\n".join(n[:10]),
                "diff": "\n".join(diff_lines[:40]),
            })

    return diffs if diffs else [{"section": section or "(all)", "diff": "No textual changes detected."}]


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_response(resp: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Minimal JSON-schema-ish validation without jsonschema library."""
    errors = []
    if "decision" not in resp:
        errors.append("Missing required field: decision")
    elif resp["decision"] not in {"Covered", "RequiresPA", "NotCovered", "Ambiguous"}:
        errors.append(f"Invalid decision value: {resp['decision']}")
    if "validity_window" not in resp:
        errors.append("Missing required field: validity_window")
    elif "effective_from" not in resp["validity_window"]:
        errors.append("validity_window missing effective_from")
    if "evidence" not in resp:
        errors.append("Missing required field: evidence")
    elif not isinstance(resp["evidence"], list):
        errors.append("evidence must be an array")
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# GCP pipeline (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _gcp_query(question: str, k: int = 8) -> Dict[str, Any]:
    """Full GCP pipeline: Vertex RAG → Gemini rerank → Gemini generate → ground."""
    try:
        from config_env import load_env
        load_env()
    except ImportError:
        pass

    from adapters.gcp_adapters import (
        VertexRagRetriever, GeminiGenerator, SimpleGroundingChecker
    )

    project  = os.environ["PROJECT_ID"]
    location = os.environ.get("LOCATION", "us-central1")
    corpus   = os.environ.get("RAG_CORPUS", "uhc-coverage-change-corpus")
    model    = os.environ.get("RERANK_MODEL", "gemini-1.5-pro")

    retriever = VertexRagRetriever(project, location, corpus)
    generator = GeminiGenerator(project, location, model)
    grounder  = SimpleGroundingChecker()

    hits = retriever.search(question, k=k)

    system_prompt = (
        "You are a healthcare coverage expert. Answer based ONLY on the provided policy excerpts. "
        "Return a JSON object with keys: decision (Covered|RequiresPA|NotCovered|Ambiguous), "
        "validity_window ({effective_from, effective_to}), evidence (array of {policy_id, section, page, quote}), "
        "changes (array of {section, old, new} if applicable), notes (array of strings). "
        "If you cannot determine, set decision to Ambiguous."
    )
    context = "\n\n---\n\n".join(
        f"[{h['policy_id']} | {h.get('version_date','')} | {h.get('section','')} | p.{h.get('page','')}]\n{h['text']}"
        for h in hits
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nRespond with JSON only."

    gen_resp = generator.generate(system_prompt, user_prompt)
    raw_text = gen_resp.get("text", "")

    # Extract JSON from markdown fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw_text)
    json_str = match.group(1) if match else raw_text

    try:
        resp = json.loads(json_str)
    except json.JSONDecodeError:
        resp = {
            "decision": "Ambiguous",
            "validity_window": {"effective_from": "unknown", "effective_to": None},
            "evidence": _build_evidence(hits),
            "notes": [f"JSON parse error; raw: {raw_text[:200]}"],
        }

    ground_score = grounder.score(raw_text, hits)
    resp.setdefault("notes", [])
    resp["notes"].append(f"GroundingScore: {ground_score:.2f}")
    resp["notes"].append(f"Model: {model}")
    resp["notes"].append(f"RetrievedChunks: {len(hits)}")

    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Public query function (used by eval runner)
# ─────────────────────────────────────────────────────────────────────────────

_chunks_cache: Optional[List[Dict[str, Any]]] = None


def query(question: str, mode: str = "local", k: int = 8,
          old_version: Optional[str] = None,
          new_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve → (optionally diff) → build response.
    mode='local': BM25 over local JSONL (no GCP needed).
    mode='gcp':   Vertex AI RAG + Gemini generation.
    """
    if mode == "gcp":
        return _gcp_query(question, k=k)

    # ── local mode ──────────────────────────────────────────────────────────
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = _load_chunks(CHUNKS_DIR)

    hits = local_retrieve(question, _chunks_cache, k=k)

    dense_active = _get_embedder() is not None
    retrieval_label = (
        f"dense={DENSE_WEIGHT}+bm25={BM25_WEIGHT}+MMR" if dense_active else "bm25+MMR"
    )

    # ── version diffs (always computed locally) ──────────────────────────────
    changes: List[Dict[str, str]] = []
    if old_version and new_version and hits:
        policy_id = hits[0].get("policy_id", "UHC-PA-COMM")
        diffs = compute_diff(policy_id, old_version, new_version)
        for d in diffs[:3]:
            if "diff" in d and "No textual" not in d["diff"]:
                changes.append({
                    "section": d["section"],
                    "old": d.get("old", "")[:200],
                    "new": d.get("new", "")[:200],
                })

    # ── grounding helper (shared by both paths) ───────────────────────────────
    def _ground_score(answer_text: str) -> float:
        ev_tokens: set = set()
        for h in hits:
            ev_tokens |= set(re.findall(r"[a-z0-9\-]{3,}", h.get("text", "").lower()))
        sents = re.split(r"(?<=[.!?])\s+", answer_text.strip())
        if not sents or not ev_tokens:
            return 0.0
        supported = sum(
            1 for s in sents
            if len(set(re.findall(r"[a-z0-9\-]{3,}", s.lower())) & ev_tokens)
               / max(1, len(re.findall(r"[a-z0-9\-]{3,}", s.lower()))) >= 0.15
        )
        return supported / max(1, len(sents))

    # ── Gemini generation ─────────────────────────────────────────────────────
    gen_resp = _gemini_generate(question, hits)
    if gen_resp is not None:
        gen_resp.setdefault("changes", changes or gen_resp.get("changes", []))
        gen_resp.setdefault("notes", [])
        # Compute grounding score over the raw quoted evidence in the response
        answer_text = " ".join(
            e.get("quote", "") for e in gen_resp.get("evidence", [])
        )
        ground = _ground_score(answer_text)
        gen_resp["notes"] = [
            n for n in gen_resp["notes"]
            if not n.startswith("GroundingScore") and not n.startswith("Mode")
        ]
        gen_resp["notes"].insert(0, f"Mode: local-hybrid ({retrieval_label}+Gemini)")
        gen_resp["notes"].append(f"GroundingScore: {ground:.2f}")
        gen_resp["notes"].append(f"Model: {GEMINI_LOCAL_MODEL}")
        gen_resp["notes"].append(f"RetrievedChunks: {len(hits)}")
        return gen_resp

    # ── rule-based fallback (no API key or generation error) ─────────────────
    decision = _infer_decision(question, hits)
    evidence = _build_evidence(hits)
    validity = _build_validity(hits)

    return {
        "decision": decision,
        "validity_window": validity,
        "evidence": evidence,
        "changes": changes,
        "notes": [
            f"Mode: local-{retrieval_label} (rule-based; set GEMINI_API_KEY for LLM generation)",
            f"RetrievedChunks: {len(hits)}",
            f"TopScore: {hits[0]['score'] if hits else 0}",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


DECISION_COLORS = {
    "RequiresPA":  "33",   # yellow
    "Covered":     "32",   # green
    "NotCovered":  "31",   # red
    "Ambiguous":   "35",   # magenta
}


def pretty_print(resp: Dict[str, Any], question: str) -> None:
    decision = resp.get("decision", "?")
    color    = DECISION_COLORS.get(decision, "37")

    print()
    print(_color("━" * 70, "90"))
    print(_color(f"  QUESTION:  {question}", "1"))
    print(_color("━" * 70, "90"))
    print(f"  Decision:  {_color(decision, color + ';1')}")
    vw = resp.get("validity_window", {})
    print(f"  Effective: {vw.get('effective_from','?')}  →  {vw.get('effective_to') or 'present'}")
    print()

    evidence = resp.get("evidence", [])
    if evidence:
        print(_color("  Evidence:", "1"))
        for i, ev in enumerate(evidence[:4], 1):
            print(f"  [{i}] {ev.get('policy_id')}  |  {ev.get('version_date','')}  |  "
                  f"§{ev.get('section','')}  p.{ev.get('page','')}")
            quote = (ev.get("quote") or "")[:160].replace("\n", " ")
            print(f"      \"{_color(quote, '36')}\"")
        print()

    changes = resp.get("changes", [])
    if changes:
        print(_color("  Version Changes Detected:", "33;1"))
        for ch in changes[:3]:
            print(f"  § {ch.get('section')}")
            for line in ch.get("diff", ch.get("new", ""))[:300].splitlines():
                prefix = line[:1]
                clr = "32" if prefix == "+" else ("31" if prefix == "-" else "90")
                print(f"    {_color(line, clr)}")
        print()

    notes = resp.get("notes", [])
    if notes:
        print(_color("  Notes: " + "  |  ".join(notes), "90"))

    ok, errs = validate_response(resp)
    if not ok:
        print(_color(f"  SCHEMA ERRORS: {errs}", "31"))

    print(_color("━" * 70, "90"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_query(args: argparse.Namespace) -> None:
    t0 = time.time()
    resp = query(
        args.question,
        mode=args.mode,
        k=args.top_k,
        old_version=args.old_version,
        new_version=args.new_version,
    )
    latency_ms = (time.time() - t0) * 1000
    resp.setdefault("notes", []).append(f"Latency: {latency_ms:.0f}ms")

    if args.json:
        print(json.dumps(resp, indent=2))
    else:
        pretty_print(resp, args.question)


def cmd_diff(args: argparse.Namespace) -> None:
    diffs = compute_diff(args.policy, args.old, args.new, section=args.section)
    if args.json:
        print(json.dumps(diffs, indent=2))
        return

    print()
    print(_color(f"  DIFF: {args.policy}  {args.old} → {args.new}", "1"))
    if args.section:
        print(f"  Section filter: {args.section}")
    print(_color("━" * 70, "90"))
    for d in diffs:
        if "error" in d:
            print(_color(f"  Error: {d['error']}", "31"))
            continue
        print(f"\n  § {_color(d['section'], '1')}")
        for line in d.get("diff", "").splitlines():
            prefix = line[:1]
            clr = "32" if prefix == "+" else ("31" if prefix == "-" else "90")
            print(f"    {_color(line, clr)}")
    print()


def cmd_eval(args: argparse.Namespace) -> None:
    eval_script = ROOT / "eval" / "run_eval.py"
    os.environ["QUERY_MODE"] = args.mode
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_eval", eval_script)
    mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(mod)                    # type: ignore[union-attr]
    mod.main()


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a canned demo showing all system capabilities."""
    demo_questions = [
        ("Does rituximab require prior authorization for UHC commercial plans?", None, None),
        ("Is bariatric surgery covered without prior authorization?", None, None),
        ("What changed for prior auth requirements between Jan 2025 and Oct 2025?",
         "1-1-2025", "10-1-2025"),
        ("What are the prior auth requirements for cardiac genetics testing?", None, None),
    ]
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = _load_chunks(CHUNKS_DIR)

    model_ready = _get_embedder() is not None
    retrieval_label = "dense+bm25+MMR" if model_ready else "bm25-only"
    print(_color("\n  UHC Change Intelligence — DEMO RUN", "1;34"))
    print(_color(f"  Mode: {args.mode}  |  Retrieval: {retrieval_label}  |  Chunks: {len(_chunks_cache)}", "90"))

    for question, old_v, new_v in demo_questions:
        t0 = time.time()
        resp = query(question, mode=args.mode, k=8, old_version=old_v, new_version=new_v)
        resp.setdefault("notes", []).append(f"Latency: {(time.time()-t0)*1000:.0f}ms")
        pretty_print(resp, question)
        time.sleep(0.2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UHC Change Intelligence CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["local", "gcp"], default="local",
                        help="Retrieval backend (default: local)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    sub = parser.add_subparsers(dest="command")

    # query
    p_query = sub.add_parser("query", help="Ask a coverage question")
    p_query.add_argument("question", help="Natural language question")
    p_query.add_argument("--top-k", type=int, default=8)
    p_query.add_argument("--versions", nargs=2, metavar=("OLD", "NEW"),
                         dest="versions", help="e.g. --versions 1-1-2025 10-1-2025")
    p_query.set_defaults(func=cmd_query)

    # diff
    p_diff = sub.add_parser("diff", help="Show section diffs between versions")
    p_diff.add_argument("--policy",  required=True, help="Policy ID e.g. UHC-PA-COMM")
    p_diff.add_argument("--old",     required=True, help="Old version e.g. 1-1-2025")
    p_diff.add_argument("--new",     required=True, help="New version e.g. 10-1-2025")
    p_diff.add_argument("--section", default=None, help="Filter to a specific section")
    p_diff.set_defaults(func=cmd_diff)

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation on gold questions")
    p_eval.set_defaults(func=cmd_eval)

    # demo
    p_demo = sub.add_parser("demo", help="Run a canned end-to-end demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    # Propagate --versions to query sub-command
    if args.command == "query":
        args.old_version = args.versions[0] if args.versions else None
        args.new_version = args.versions[1] if args.versions else None

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
