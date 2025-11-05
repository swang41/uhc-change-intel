# UHC Coverage & Prior-Auth Change Intelligence (Ultra-Light MVP)

This is an ultra-light MVP showing **version-aware RAG** on UnitedHealthcare (UHC) Commercial prior-authorization/coverage PDFs for **2025** (e.g., Jan vs Oct versions).

## What this MVP demonstrates
- Retrieval-Augmented Generation (RAG) with **citations (policy/section/page)**.
- **Version-aware** responses with **diff snippets** for changed sections.
- **Grounding gate** (refuse if support score below threshold).
- **Strict JSON** responses validated against a schema.
- A small **evaluation set** with hit-rate, grounded accuracy, latency, and cost estimates.

## Quick Start (placeholder)
1. Create and export GCP credentials with access to Vertex AI & Cloud Storage.
2. Put 5 PDFs in GCS (Jan-2025 PA, Oct-2025 PA, 3 Commercial policies).
3. Fill `docs/manifest_template.csv` with rows for those PDFs.
4. Run `scripts/ingest.py` to parse, chunk, embed, and load corpus (RAG Managed DB).
5. Use `eval/run_eval.py` to run a small evaluation (15–20 questions).

> NOTE: This starter includes **pseudocode** placeholders for GCP-specific calls. Replace TODOs with actual Vertex SDK calls when you're ready.

## Repo structure
