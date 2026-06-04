# UHC Coverage & Prior-Auth Change Intelligence

A **version-aware RAG system** that ingests UnitedHealthcare (UHC) Commercial prior-authorization and coverage policy PDFs, tracks changes across versions (e.g., Jan 2025 → Oct 2025), and answers natural language coverage questions with grounded, citeable evidence.

> **Demo-ready with no GCP credentials required** — set `GEMINI_API_KEY` and run `python app/app.py demo` to see the full pipeline: hybrid retrieval → Gemini generation → grounding check → schema-validated JSON.

---

## What This Demonstrates

| Capability | Details |
|---|---|
| **Version-aware RAG** | Chunks tagged with `policy_id`, `version_date`, `section`, `page` — every answer cites its source |
| **Hybrid retrieval** | Dense (sentence-transformers `all-MiniLM-L6-v2`) + BM25 (proper IDF), fused via **Reciprocal Rank Fusion**, diversity via **MMR** |
| **LLM generation** | Gemini produces structured JSON from retrieved context; works locally with `GEMINI_API_KEY` or on GCP with Vertex AI |
| **Grounding gate** | Token-overlap heuristic scores each answer sentence against evidence; low-support responses flagged |
| **Diff tracking** | Unified-diff of parsed sections between versions detects exact wording changes |
| **Strict JSON output** | All responses validated against `schemas/response.schema.json` (decision, evidence, validity window) |
| **Eval harness** | 15-question gold set; reports hit-rate, grounded accuracy, p95 latency, $/100 queries |

---

## Architecture

```
PDF (GCS)
   │
   ▼
scripts/ingest.py
   ├── pypdf / pdfminer  →  section detection  →  smart chunking (~900 tok, 15% overlap)
   └── output: data/parsed/*.jsonl   data/chunks/*.jsonl
                    │
                    ▼
          scripts/rag_loader.py  ──►  Vertex RAG Managed DB  (GCP mode only)

                    │
        ┌───────────┴────────────┐
        │  local mode            │  GCP mode
        │                        │
        │  Dense retrieval        │  Vertex RAG (dense)
        │  all-MiniLM-L6-v2      │  + keyword boost
        │  cosine similarity      │  + domain aliases
        │        +               │        +
        │  BM25 (proper IDF)      │  MMR diversity
        │        │               │        │
        │   RRF fusion           │   LLM rerank
        │   (rank⁻¹ 60/40)       │   (Gemini 1.5 Pro)
        │        │               │        │
        │   MMR diversity        │        │
        │   (λ = 0.7)            │        │
        └───────────┬────────────┘
                    │
                    ▼
           Gemini generate
           (google-generativeai  or  Vertex AI)
                    │
                    ▼
           Grounding check  (token-overlap, sentence-level)
                    │
                    ▼
        JSON response  (schema-validated)
      + diff snippets  (local JSONL unified-diff)
```

---

## Hybrid Retrieval — Verified Design

The local retrieval pipeline has three stages, all verified with a unit test:

### 1. Dual Scoring

| Signal | Method | Strength |
|---|---|---|
| **Dense** | `all-MiniLM-L6-v2` cosine similarity (384-dim) | Catches paraphrases — "PA required" ≈ "prior authorization needed" |
| **BM25** | TF-IDF with proper corpus IDF over all 392 chunks | Exact CPT/HCPCS codes, drug names (J9312, rituximab) |

### 2. Reciprocal Rank Fusion

Rather than tuning a linear interpolation weight, RRF fuses by rank position:

```
score(chunk) = 0.6 × (1 / (rank_dense + 60))
             + 0.4 × (1 / (rank_bm25  + 60))
```

This is robust to score-scale differences between the two signals.

### 3. MMR Diversity (λ = 0.7)

Applied to the top-40 RRF candidates before the final top-k cut:

```
MMR(c) = λ · sim(query, c) − (1−λ) · max_{s∈selected} sim(s, c)
```

Ensures results span multiple policies and sections rather than returning five chunks from the same paragraph.

### Verified Results (unit test with deterministic fake embedder)

```
Rank  Score   Dense   BM25   Policy / Version
   1  0.01339  1.000  0.517  UHC-PA-COMM / 1-1-2025        ← high on both
   7  0.01293  0.917  0.534  UHC-DRUG-RITUXIMAB / 10-1-2025 ← keyword match
   2  0.01007  0.875  0.000  UHC-MED-GENETICS-CARDIAC       ← semantic only
```

Dense-only hits (BM25=0) and BM25-dominated hits coexist, confirming both signals contribute. No policy appears more than twice (MMR enforced).

---

## Quick Start

### 1 — Launch the web UI

```bash
git clone https://github.com/swang41/uhc-change-intel
cd uhc-change-intel
pip install -r requirements.txt

# Add your Gemini API key (free at aistudio.google.com)
echo "GEMINI_API_KEY=your_key_here" > config/.env

python app/server.py        # open http://localhost:5000
```

> Without `GEMINI_API_KEY` the server still works — rule-based decisions are returned instead of LLM-generated answers.

### 2 — CLI demo (no browser needed)

```bash
# local hybrid retrieval + Gemini generation (default)
python app/app.py query "Does rituximab require prior authorization?"

# with version-diff
python app/app.py query "What changed for bariatric surgery?" \
    --versions 1-1-2025 10-1-2025

# raw JSON output
python app/app.py --json query "Is arthroplasty subject to PA?"
```

### 3 — View policy diffs

```bash
python app/app.py diff --policy UHC-PA-COMM --old 1-1-2025 --new 10-1-2025

# filter to one section
python app/app.py diff --policy UHC-PA-COMM \
    --old 1-1-2025 --new 10-1-2025 --section "Bariatric surgery"
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
  Effective: 2025-10-01  →  present

  Evidence:
  [1] UHC-DRUG-RITUXIMAB  |  10-1-2025  |  §Coverage Rationale  p.1
      "Prior authorization is required for rituximab injections for
       intravenous infusion for non-oncology conditions ..."
  [2] UHC-PA-COMM  |  10-1-2025  |  §Injectable medications  p.4
      "Rituximab (Rituxan®) J9312 — prior authorization required ..."

  Notes: Mode: local-hybrid  |  RetrievedChunks: 8  |  GroundingScore: 0.87
         Model: gemini-2.0-flash  |  Latency: 1240ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Evaluation Results (local hybrid mode)

| Metric | Value |
|---|---|
| Retrieval hit-rate | 100% (15/15) |
| Grounded accuracy | 60% (rule-based) → higher with Gemini generation |
| p95 latency | < 70 ms retrieval / ~1.5 s with generation |
| Cost / 100 queries | ~$0.01 (Gemini Flash local) / ~$0.35 (Vertex AI GCP) |

*Full results in `eval/results.csv` after running `python eval/run_eval.py`.*

---

## Repo Structure

```
uhc-change-intel/
├── app/
│   └── app.py                   CLI — query / diff / eval / demo commands
├── config/
│   ├── .env.example             Credentials template (GCP + Gemini API key)
│   └── config_env.py            dotenv loader with priority overrides
├── data/
│   ├── parsed/                  11 × JSONL — section-level extracted text
│   └── chunks/                  11 × JSONL — ~900-token chunks with metadata
├── docs/
│   ├── manifest.csv             11 PDFs (policy_id, version, GCS path, SHA256)
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

| Layer | Technology |
|---|---|
| **Cloud (GCP mode)** | Vertex AI RAG Managed DB, Gemini 1.5 Pro, Cloud Storage |
| **Dense embeddings** | sentence-transformers `all-MiniLM-L6-v2` (384-dim, 22 MB) |
| **BM25** | Custom implementation with proper corpus IDF |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Diversity** | Maximal Marginal Relevance (MMR, λ=0.7) |
| **LLM generation** | Gemini 2.0 Flash via `google-generativeai` (local) or Vertex AI (GCP) |
| **Grounding** | Token-overlap heuristic, sentence-level |
| **PDF parsing** | pypdf, pdfminer.six, tiktoken |
| **Schema validation** | Custom JSON Schema validator |
| **Diff** | Python `difflib` unified diff on JSONL sections |

---

## Design Decisions

**Why version-aware metadata?**  Healthcare policies change quarterly. Tagging every chunk with `policy_id`, `version_date`, and `effective_from` enables "what changed?" as a first-class operation, not an afterthought.

**Why hybrid retrieval with RRF?**  Dense embeddings (`all-MiniLM-L6-v2`) catch semantic paraphrases — "PA required" matches "prior authorization needed" — while BM25 with proper IDF nails exact CPT/HCPCS codes and drug names (J9312, rituximab). RRF fuses by rank position rather than score magnitude, so it requires no threshold tuning across different score scales. MMR then ensures the final top-k spans multiple policies instead of repeating the same paragraph.

**Why Gemini generation in local mode?**  The `google-generativeai` SDK needs only an API key (free tier available), making the full RAG pipeline — retrieve → generate → ground → validate — runnable on any laptop without GCP project setup. The same structured prompt and JSON schema are used in both local and GCP modes so the output format is identical.

**Why strict JSON schema?**  Downstream consumers (billing systems, clinical workflows) need machine-readable, predictable output — not free-text summaries.
