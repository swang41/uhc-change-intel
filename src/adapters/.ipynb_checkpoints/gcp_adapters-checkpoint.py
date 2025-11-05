# src/gcp/adapters.py
from __future__ import annotations
import os, re, json, difflib
from typing import Any, Dict, List, Optional

from vertexai.generative_models import GenerativeModel, Part
import vertexai

from gcp_rag.hybrid_retriever_reranker import hybrid_retrieve, rerank_chunks

from core.interfaces import Retriever, Generator, GroundingChecker, DiffClient

# ---------- Retriever adapter (Vertex RAG + hybrid + LLM rerank) ----------
class VertexRagRetriever(Retriever):
    def __init__(self, project_id: str, location: str, corpus: str, top_candidates: int = 5):
        self.project_id = project_id
        self.location = location
        self.corpus = corpus
        self.top_candidates = top_candidates
        vertexai.init(project=project_id, location=location)
        # let the underlying module read env vars for CORPUS if needed

    def search(self, query: str, k: int = 8, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Hybrid retrieve (dense + keyword + MMR), then rerank with LLM and return top-k
        candidates = hybrid_retrieve(query)
        ranked = rerank_chunks(query, candidates, top_n=max(k, self.top_candidates))
        # shape: return dicts with required keys
        out = []
        for r in ranked[:k]:
            out.append({
                "text": r["text"],
                "score": float(r["score_final"]),
                "score_dense": float(r["score_dense"]),
                "score_llm": float(r["score_llm"]),
                "uri": r.get("uri"),
                "policy_id": r.get("policy_id"),
                "version_date": r.get("version_date"),
                "effective_from": r.get("effective_from"),
                "section": r.get("section"),
                "page": r.get("page"),
                "chunk_id": r.get("chunk_id"),
            })
        return out

# ---------- Generator adapter (Gemini) ----------
class GeminiGenerator(Generator):
    def __init__(self, project_id: str, location: str, model: str = "gemini-1.5-pro", temperature: float = 0.0):
        self.model_name = model
        self.temperature = temperature
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model)

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        parts = [Part.from_text(system_prompt + "\n\n" + user_prompt)]
        resp = self.model.generate_content(
            parts,
            generation_config={"temperature": self.temperature}
        )
        return {"text": (resp.text or ""), "raw": resp}

# ---------- Grounding checker (heuristic sentence support) ----------
class SimpleGroundingChecker(GroundingChecker):
    def __init__(self, min_overlap: float = 0.15):
        self.min_overlap = min_overlap

    @staticmethod
    def _tokens(s: str) -> set:
        return set(t for t in re.findall(r"[A-Za-z0-9\-]+", (s or "").lower()) if len(t) > 2)

    def score(self, answer_text: str, evidences: List[Dict[str, Any]]) -> float:
        if not answer_text or not evidences:
            return 0.0
        ev_text = "\n".join(e.get("text", "") for e in evidences)
        ev_tokens = self._tokens(ev_text)
        if not ev_tokens:
            return 0.0
        sents = re.split(r"(?<=[.!?])\s+", answer_text.strip())
        if not sents:
            return 0.0
        supported = 0
        for s in sents:
            at = self._tokens(s)
            if not at:
                continue
            overlap = len(at & ev_tokens) / max(1, len(at))
            if overlap >= self.min_overlap:
                supported += 1
        return supported / max(1, len(sents))

# ---------- Diff client (local JSONL compare of parsed sections) ----------
class LocalJsonlDiffClient(DiffClient):
    """
    Looks in data/parsed/{policy_id}_{version}.jsonl, groups by section,
    and computes line-level diffs for the requested section.
    """
    def __init__(self, parsed_dir: str = "data/parsed"):
        self.parsed_dir = parsed_dir

    def _load_sections(self, policy_id: str, version: str) -> Dict[str, str]:
        import os
        path = os.path.join(self.parsed_dir, f"{policy_id}_{version.replace('/','-')}.jsonl")
        sections: Dict[str, List[str]] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    title = (rec.get("section") or "").strip()
                    sections.setdefault(title, []).append(rec.get("text",""))
        except FileNotFoundError:
            return {}
        # concatenate text per section
        return {k: "\n".join(v) for k, v in sections.items()}

    def section_diffs(self, policy_id: str, old_version: str, new_version: str, section: str) -> List[Dict[str, str]]:
        old_map = self._load_sections(policy_id, old_version)
        new_map = self._load_sections(policy_id, new_version)
        o = (old_map.get(section) or "").splitlines()
        n = (new_map.get(section) or "").splitlines()
        diff = list(difflib.unified_diff(o, n, lineterm="", fromfile=f"{old_version}", tofile=f"{new_version}"))
        return [{"section": section, "diff": "\n".join(diff)}]
