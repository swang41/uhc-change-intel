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

CHUNKS_DIR  = ROOT / "data" / "chunks"
PARSED_DIR  = ROOT / "data" / "parsed"
SCHEMA_FILE = ROOT / "schemas" / "response.schema.json"

# ─────────────────────────────────────────────────────────────────────────────
# Local BM25-style retriever (no GCP required)
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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\-]+", (text or "").lower())


def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    freq: Dict[str, int] = {}
    for t in doc_tokens:
        freq[t] = freq.get(t, 0) + 1
    dl = len(doc_tokens)
    score = 0.0
    for t in set(query_tokens):
        tf = freq.get(t, 0)
        if tf == 0:
            continue
        idf = math.log(1 + 1)  # simplified; corpus too small for real IDF
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avgdl)))
        score += idf * tf_norm
    return score


def local_retrieve(query: str, chunks: List[Dict[str, Any]],
                   k: int = 8) -> List[Dict[str, Any]]:
    """BM25-style retrieval over local chunk store."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    doc_tokens_list = [_tokenize(c.get("text", "")) for c in chunks]
    avgdl = sum(len(t) for t in doc_tokens_list) / max(1, len(doc_tokens_list))

    scored = []
    for i, chunk in enumerate(chunks):
        score = _bm25_score(q_tokens, doc_tokens_list[i], avgdl)
        if score > 0:
            scored.append((score, i))

    scored.sort(reverse=True)

    # MMR-style dedup: limit to 2 chunks per policy+version
    seen: Dict[str, int] = {}
    results = []
    for score, idx in scored:
        chunk = chunks[idx]
        key = f"{chunk.get('policy_id')}_{chunk.get('version_date')}"
        if seen.get(key, 0) >= 2:
            continue
        seen[key] = seen.get(key, 0) + 1
        result = dict(chunk)
        result["score"] = round(score, 4)
        results.append(result)
        if len(results) >= k:
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Response builder (local mode — structured without LLM)
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

    decision  = _infer_decision(question, hits)
    evidence  = _build_evidence(hits)
    validity  = _build_validity(hits)
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

    resp = {
        "decision": decision,
        "validity_window": validity,
        "evidence": evidence,
        "changes": changes,
        "notes": [
            f"Mode: local-BM25",
            f"RetrievedChunks: {len(hits)}",
            f"TopScore: {hits[0]['score'] if hits else 0}",
        ],
    }
    return resp


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

    print(_color("\n  UHC Change Intelligence — DEMO RUN", "1;34"))
    print(_color(f"  Mode: {args.mode}  |  Chunks loaded: {len(_chunks_cache)}", "90"))

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
