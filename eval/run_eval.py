"""
Evaluation runner (pseudocode):
- Reads gold questions CSV
- Calls query() to get JSON response
- Computes retrieval hit-rate, grounded accuracy, p95 latency, and $/100 queries (rough estimate)
Replace TODOs with real calls.
"""

import csv, json, time, statistics
from pathlib import Path

GOLD = "eval/gold_questions.csv"  # copy template and fill
RESULTS = "eval/results_gcp.csv"

def query_system(question: str) -> dict:
    # TODO: Implement: retrieve -> generate -> grounding -> JSON validation
    # Return a dict matching schemas/response.schema.json
    return {
        "decision": "RequiresPA",
        "validity_window": {"effective_from":"2025-01-01","effective_to": None},
        "evidence": [{"policy_id":"UHC-PA-2025","section":"Cardiology","page":5,"quote":"..."}],
        "changes": [{"section":"Cardiology","old":"...","new":"..."}],
        "notes": ["SupportScore: 0.82","Model: gemini-flash-2.x"]
    }

def main():
    gold = []
    with open(GOLD, "r") as f:
        gold = list(csv.DictReader(f))

    latencies = []
    correct = 0
    retrieved = 0
    cost_total = 0.0

    rows = []
    for i, g in enumerate(gold):
        t0 = time.time()
        resp = query_system(g["question"])
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        # Grounded accuracy: decision matches and at least one evidence present
        is_correct = (resp.get("decision") == g.get("expected_decision")) and bool(resp.get("evidence"))
        if is_correct:
            correct += 1

        # Retrieval hit-rate (proxy): if evidence section/page match expected (loose)
        ev = resp.get("evidence", [{}])[0]
        if ev.get("expected_section", g.get("expected_section")):
            retrieved += 1

        # Rough cost estimate (placeholder fixed per-call for now)
        cost_total += 0.002  # $0.002 per query placeholder

        rows.append({
            "question_id": i,
            "latency_ms": round(latency, 2),
            "decision": resp.get("decision"),
            "correct": int(is_correct)
        })

    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies or [0])
    hit_rate = retrieved / max(1, len(gold))
    acc = correct / max(1, len(gold))
    cost_per_100 = cost_total / max(1, len(gold)) * 100.0

    # Write results CSV
    with open(RESULTS, "w") as w:
        w.write("question_id,latency_ms,decision,correct\n")
        for r in rows:
            w.write(f"{r['question_id']},{r['latency_ms']},{r['decision']},{r['correct']}\n")

    # Print summary for Metrics.md
    print("SUMMARY")
    print(f"Hit-rate (proxy): {hit_rate:.2f}")
    print(f"Grounded accuracy: {acc:.2f}")
    print(f"p95 latency (ms): {p95:.0f}")
    print(f"$ per 100 queries (rough): ${cost_per_100:.3f}")

if __name__ == "__main__":
    main()
