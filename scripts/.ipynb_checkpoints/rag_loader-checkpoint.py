# rag_loader.py
# pip install --upgrade google-cloud-aiplatform
# gcloud auth application-default login
# gcloud config set project YOUR_PROJECT_ID

import csv
import os
from typing import List

import vertexai
from vertexai import rag

# --- REQUIRED: set your project/region/corpus name ---
PROJECT_ID = os.environ.get("PROJECT_ID", "YOUR_PROJECT_ID")
LOCATION   = os.environ.get("LOCATION",   "us-central1")  # or us-east4, etc.
CORPUS_DISPLAY_NAME = os.environ.get("CORPUS_NAME", "uhc-coverage-change-corpus")

# Path to your manifest.csv with a "gcs_uri" column (one row per PDF)
MANIFEST_CSV = "manifest.csv"

# --- optional knobs ---
# Retrieval strategy: KNN is fine for <10k files (perfect recall). Use ANN when scaling up.
USE_ANN = False  # set True when you exceed ~10k rag files

# Chunking config the RAG Engine will apply during import
CHUNK_SIZE   = 900   # ~tokens; API accepts integers; adjust if you see overly small/large chunks
CHUNK_OVERLAP = 135  # ~15%

# Throttle embeddings requests to stay under quota (adjust as needed)
MAX_EMBED_REQS_PER_MIN = 1500

def read_gcs_uris(manifest_csv: str) -> List[str]:
    uris = []
    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = (row.get("gcs_uri") or "").strip()
            if uri.startswith("gs://"):
                uris.append(uri)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for u in uris:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq

def main():
    assert PROJECT_ID != "YOUR_PROJECT_ID", "Set PROJECT_ID or export env var PROJECT_ID"
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # --- Create the corpus backed by RagManagedDb ---
    if USE_ANN:
        vector_db = rag.RagManagedDb(retrieval_strategy=rag.ANN(tree_depth=2, leaf_count=500))
    else:
        vector_db = rag.RagManagedDb(retrieval_strategy=rag.KNN())

    rag_corpus = rag.create_corpus(
        display_name=CORPUS_DISPLAY_NAME,
        backend_config=rag.RagVectorDbConfig(vector_db=vector_db),
    )
    print("Created corpus:", rag_corpus.name)

    # --- Import files from GCS; RAG Engine chunks + embeds for you ---
    gcs_uris = read_gcs_uris(MANIFEST_CSV)
    if not gcs_uris:
        raise SystemExit("No gs:// URIs found in manifest.csv (need a gcs_uri column).")

    print(f"Importing {len(gcs_uris)} PDFs ...")
    rag.import_files(
        rag_corpus.name,
        gcs_uris,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ),
        ),
        # This throttles embedding calls on Google's side to avoid 429s
        max_embedding_requests_per_min=MAX_EMBED_REQS_PER_MIN,
        # For ANN only: you can rebuild index after bulk import for best recall:
        # rebuild_ann_index=True,
    )
    print("Import submitted. (Imports run asynchronously; check console for status.)")

    # --- Quick retrieval smoke test (direct retriever, no LLM) ---
    cfg = rag.RagRetrievalConfig(top_k=3)
    resp = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
        text="As of Oct 1, 2025, what are the preferred rituximab products for non-oncology?",
        rag_retrieval_config=cfg,
    )
    print("\nSMOKE TEST — top snippets:")
    for i, hit in enumerate(resp.context.citations or []):
        print(f"[{i+1}] score={hit.score:.3f} | uri={hit.uri}")
        print((hit.content or "")[:400].strip(), "\n")

if __name__ == "__main__":
    main()
