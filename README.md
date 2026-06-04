# UHC Coverage & Prior-Auth Change Intelligence

A **version-aware RAG system** that ingests UnitedHealthcare (UHC) Commercial prior-authorization and coverage policy PDFs, tracks changes across versions (e.g., Jan 2025 → Oct 2025), and answers natural language coverage questions with grounded, citeable evidence.

> **Demo-ready with no GCP credentials required** — run `python app/app.py demo` to see the full pipeline using local BM25 retrieval over the pre-parsed policy corpus.

---

## What This Demonstrates

| Capability | Details |
|---|---|
| **Version-aware RAG** | Chunks are tagged with `policy_id`, `version_date`, `section`, and `page` — every answer cites its source |
| **Hybrid retrieval** | Dense (sentence-transformers) + BM25 with proper IDF, fused via Reciprocal Rank Fusion + MMR diversity |
| **LLM re-ranking** | Gemini 1.5 Pro scores each candidate chunk; domain-aware section boosts |
| **Diff tracking** | Unified-diff of parsed sections between policy versions detects wording changes |
| **Strict JSON output** | Responses validated against `schemas/response.schema.json` (decision, evidence, validity window) |
| **Grounding gate** | Token-overlap heuristic measures answer support; low-grounding responses flagged |
| **Offline demo mode** | Full pipeline runs locally via BM25 over pre-parsed JSONL — no cloud credentials needed |
| **Eval harness** | Gold question set; reports hit-rate, grounded accuracy, p95 latency, $/100 queries |

---

## Architecture

```
PDF (GCS)
   │
   ▼
scripts/ingest.py ──── pypdf / pdfminer ──── section detection ──── chunking (~900 tok, 15% overlap)
   │                                                                        │
   ▼                                                                        ▼
docs/manifest.csv                                              data/parsed/*.jsonl
                                                               data/chunks/*.jsonl
                                                                        │
                                                                        ▼
                                              scripts/rag_loader.py ──── Vertex RAG Managed DB
                                                                              │
                                                                              ▼
                                                                   app/app.py query "..."
                                                                              │
                                          ┌───────────────────────────────────┤
                                          │                                   │
                                     GCP mode                           local mode
                                  Vertex RAG dense                     BM25 on JSONL
                                  + keyword + MMR                           │
                                          │                                  │
                                          └──────────────┬──────────────────┘
                                                         ▼
                                                  LLM re-rank (Gemini) [GCP only]
                                                         │
                                                         ▼
                                               Gemini generate [GCP only]
                                              or rule-based infer [local]
                                                         │
                                                         ▼
                                             Grounding check (token overlap)
                                                         │
                                                         ▼
                                            JSON response (schema-validated)
                                        + diff snippets (local JSONL compare)
```

---

## Quick Start

### 1 — Run the demo (no cloud credentials needed)

```bash
git clone https://github.com/swang41/uhc-change-intel
cd uhc-change-intel
pip install -r requirements.txt

python app/app.py demo
```

### 2 — Ask a specific question

```bash
# local BM25 mode (default)
python app/app.py query "Does rituximab require prior authorization?"

# with version-diff
python app/app.py query "What changed for bariatric surgery?" \
    --versions 1-1-2025 10-1-2025

# output raw JSON
python app/app.py --json query "Is arthroplasty subject to PA?"
```

### 3 — View policy diffs

```bash
python app/app.py diff \
    --policy UHC-PA-COMM \
    --old 1-1-2025 \
    --new 10-1-2025

# filter to one section
python app/app.py diff \
    --policy UHC-PA-COMM \
    --old 1-1-2025 \
    --new 10-1-2025 \
    --section "Bariatric surgery"
```

### 4 — Run the evaluation harness

```bash
python eval/run_eval.py
# results written to eval/results.csv
```

### 5 — Full GCP pipeline (requires Vertex AI access)

```bash
cp config/.env.example config/.env
# fill in PROJECT_ID, LOCATION, RAG_CORPUS

python scripts/ingest.py          # parse & chunk PDFs
python scripts/rag_loader.py      # upload to Vertex RAG Managed DB

python app/app.py --mode gcp query "Does rituximab require prior auth?"
```

---

## Sample Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUESTION:  Does rituximab require prior authorization?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Decision:  RequiresPA
  Effective: 10-1-2025  →  present

  Evidence:
  [1] UHC-DRUG-RITUXIMAB  |  10-1-2025  |  §Prior Authorization  p.1
      "Prior authorization is required for rituximab ..."
  [2] UHC-PA-COMM  |  10-1-2025  |  §Injectable medications  p.4
      "Rituximab (Rituxan®) J9312 — prior authorization required ..."

  Notes: Mode: local-BM25  |  RetrievedChunks: 8  |  Latency: 12ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Evaluation Results (local BM25 mode)

| Metric | Value |
|---|---|
| Retrieval hit-rate | 100% (15/15) |
| Grounded accuracy | 60% (9/15) — local BM25 / no LLM |
| p95 latency | < 70 ms (local) |
| Cost / 100 queries | $0.00 (local) / ~$0.35 (GCP) |

*Full results in `eval/results.csv` after running `python eval/run_eval.py`.*

---

## Repo Structure

```
uhc-change-intel/
├── app/
│   └── app.py                   CLI — query / diff / eval / demo commands
├── config/
│   ├── .env.example             GCP credentials template
│   └── config_env.py            dotenv loader with priority overrides
├── data/
│   ├── parsed/                  11 × JSONL — section-level extracted text
│   └── chunks/                  11 × JSONL — ~900-token chunks with metadata
├── docs/
│   ├── manifest.csv             11 PDFs loaded (policy_id, version, GCS path, SHA256)
│   └── manifest_template.csv   Onboarding template
├── eval/
│   ├── gold_questions.csv       15 labelled evaluation questions
│   ├── results.csv              Generated after running eval
│   └── run_eval.py              Evaluation harness
├── schemas/
│   └── response.schema.json     JSON schema for API responses
├── scripts/
│   ├── ingest.py                PDF → JSONL parsing + chunking pipeline
│   └── rag_loader.py            Vertex RAG corpus creation + smoke test
└── src/
    ├── core/interfaces.py       Abstract base classes (Retriever, Generator, ...)
    ├── adapters/gcp_adapters.py GCP implementations
    └── gcp_rag/
        └── hybrid_retriever_reranker.py  Dense + keyword + MMR + LLM rerank
```

---

## Policy Corpus

| Policy ID | Description | Versions |
|---|---|---|
| UHC-PA-COMM | Commercial Prior Authorization list | Jan 2025, Oct 2025 |
| UHC-PA-CHANGES | Summary of PA Changes | Oct 2025 |
| UHC-DRUG-RITUXIMAB | Rituximab coverage criteria | Oct 2025 |
| UHC-DRUG-LEQVIO | Leqvio (inclisiran) coverage | Oct 2025 |
| UHC-MED-PREVENTIVE | Preventive care policy | Oct 2025 |
| UHC-MED-GENETICS-CARDIAC | Cardiac genetics testing | Oct 2025 |
| UHC-MED-MSK-ABLATION | MSK ablation procedures | Aug 2025 |
| UHC-MED-DME-REPAIR-REPLACE | DME repair & replacement | Aug 2025 |
| UHC-MED-GYN-AUB | Gynecology / AUB treatment | Oct 2025 |
| UHC-MED-IMAGING-SOS-MRI-CT | MRI/CT soft tissue sarcoma | Sep 2025 |

---

## Tech Stack

- **Cloud**: GCP Vertex AI (RAG Managed DB, Gemini 1.5 Pro), Cloud Storage
- **PDF parsing**: pypdf, pdfminer.six, tiktoken
- **Local retrieval**: BM25 (custom, no external dependencies)
- **Hybrid retrieval**: Dense + keyword + MMR + LLM re-ranking
- **Schema validation**: JSON Schema (custom validator, no jsonschema dep for local mode)
- **Diff**: Python `difflib` unified diff on parsed JSONL sections
- **Python**: 3.10+

---

## Design Decisions

**Why version-aware metadata?**  Healthcare policies change quarterly. Tagging every chunk with `policy_id`, `version_date`, and `effective_from` enables the system to answer "what changed?" as a first-class operation, not an afterthought.

**Why hybrid retrieval?**  Dense embeddings (all-MiniLM-L6-v2) capture semantic similarity — "prior authorization" matches "PA required" — while BM25 with proper IDF catches exact CPT/HCPCS codes and drug names. Reciprocal Rank Fusion merges the two rankings without needing to tune a score threshold. MMR then ensures the final top-k spans multiple policies and sections rather than repeating the same paragraph.

**Why an offline demo mode?**  Pre-parsed JSONL makes the project self-contained for interviews and demos without requiring cloud credentials or incurring API costs. The local pipeline automatically uses full hybrid search (dense + BM25 + MMR) when `sentence-transformers` is installed, and gracefully falls back to BM25-only otherwise.

**Why strict JSON schema?**  Downstream consumers (billing systems, clinical workflows) need machine-readable, predictable output — not free-text summaries.
