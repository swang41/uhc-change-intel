"""
Pre-build the local dense embedding cache so the first query is instant.

Usage:
    python scripts/build_embeddings.py

Writes:
    data/embeddings.npy          float32 matrix, shape (N, 384)
    data/embeddings_meta.json    {"n_chunks": N, "chunk_ids": [...]}
"""

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from app import _load_chunks, _load_or_build_embeddings, CHUNKS_DIR, EMBED_CACHE, EMBED_META

chunks = _load_chunks(CHUNKS_DIR)
print(f"Loaded {len(chunks)} chunks from {CHUNKS_DIR}")

matrix, chunk_ids = _load_or_build_embeddings(chunks)

if matrix is None:
    print("ERROR: sentence-transformers not installed or model unavailable.")
    print("  pip install sentence-transformers")
    sys.exit(1)

print(f"Embeddings saved to {EMBED_CACHE}  (shape {matrix.shape})")
print(f"Metadata   saved to {EMBED_META}")
