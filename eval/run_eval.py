"""
Evaluation runner for UHC Change Intelligence.
Reads gold questions from eval/gold_questions.csv, calls the query pipeline,
and reports: hit-rate, grounded accuracy, p95 latency, $/100 queries.

Usage:
    python eval/run_eval.py                     # local BM25 mode
    QUERY_MODE=gcp python eval/run_eval.py      # GCP Vertex mode
"""

import csv, json, os, re, sys, time, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))

GOLD    = ROOT / "eval" / "gold_questions.csv"
RESULTS = ROOT / "eval" / "results.csv"

# Cost estimates (rough; GCP pricing as of 2025)
COST_PER_QUERY_GCP   = 0.0035   # Gemini 1.5 Pro input+output for typical RAG call
COST_PER_QUERY_LOCAL = 0.0000   # local BM25, no API cost


def query_system(question: str, mode: str = "local") -> dict:
    """Retrieve → build response → validate schema."""
    import importlib.util
    app_path = ROOT / "app" / "app.py"
    spec = importlib.util.spec_from_file_location("app", app_path)
    mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(mod)                  # type: ignore[union-attr]
    return mod.query(question, mode=mode)


def main():
    mode = os.environ.get("QUERY_MODE", "local")
    cost_per_q = COST_PER_QUERY_GCP if mode == "gcp" else COST_PER_QUERY_LOCAL

    if not GOLD.exists():
        print(f"Gold questions file not found: {GOLD}")
        print("Run: cp eval/gold_questions.csv.example eval/gold_questions.csv  (or create it)")
        sys.exit(1)

    gold = []
    with open(GOLD, encoding="utf-8") as f:
        gold = list(csv.DictReader(f))

    if not gold:
        print("gold_questions.csv is empty.")
        sys.exit(1)

    print(f"\nEvaluating {len(gold)} questions in [{mode}] mode ...\n")

    latencies = []
    correct   = 0
    retrieved = 0
    rows      = []

    for i, g in enumerate(gold):
        question = g["question"]
        expected_decision = g.get("expected_decision", "")
        expected_section  = g.get("expected_section", "")

        t0 = time.time()
        try:
            resp = query_system(question, mode=mode)
        except Exception as e:
            resp = {
                "decision": "Ambiguous",
                "validity_window": {"effective_from": "unknown", "effective_to": None},
                "evidence": [],
                "notes": [f"ERROR: {e}"],
            }
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        # Grounded accuracy: decision matches expected AND has evidence
        is_correct = (
            bool(expected_decision) and
            resp.get("decision") == expected_decision and
            bool(resp.get("evidence"))
        )
        if is_correct:
            correct += 1

        # Retrieval hit-rate: evidence mentions expected section/policy OR question keywords in quotes
        ev_sections = [e.get("section", "").lower()   for e in resp.get("evidence", [])]
        ev_policies = [e.get("policy_id", "").lower() for e in resp.get("evidence", [])]
        ev_combined = " ".join(ev_sections + ev_policies)
        section_hit = bool(expected_section and expected_section.lower() in ev_combined)
        q_words     = set(re.findall(r"[a-z]{4,}", question.lower()))
        ev_text     = " ".join(e.get("quote", "").lower() for e in resp.get("evidence", []))
        keyword_hit = len(q_words & set(re.findall(r"[a-z]{4,}", ev_text))) >= 2
        hit = section_hit or keyword_hit or not expected_section
        if hit:
            retrieved += 1

        decision = resp.get("decision", "?")
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}] {status} {decision:<12}  {latency_ms:6.0f}ms  {question[:60]}")

        rows.append({
            "question_id":      i + 1,
            "question":         question,
            "expected_decision":expected_decision,
            "predicted_decision":decision,
            "correct":          int(is_correct),
            "hit":              int(hit),
            "latency_ms":       round(latency_ms, 2),
        })

    # ── metrics ──────────────────────────────────────────────────────────────
    n         = max(1, len(gold))
    p95       = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies or [0])
    hit_rate  = retrieved / n
    acc       = correct / n
    cost_100  = cost_per_q * 100

    # ── write results CSV ────────────────────────────────────────────────────
    with open(RESULTS, "w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(w, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*50}")
    print(f"  Retrieval hit-rate:    {hit_rate:.0%}  ({retrieved}/{n})")
    print(f"  Grounded accuracy:     {acc:.0%}  ({correct}/{n})")
    print(f"  p95 latency:           {p95:.0f} ms")
    print(f"  Est. cost / 100 q:     ${cost_100:.3f}  ({mode} mode)")
    print(f"  Results written to:    {RESULTS.relative_to(ROOT)}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
