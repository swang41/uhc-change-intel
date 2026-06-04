"""
Flask web UI for UHC Change Intelligence.

Usage:
    python app/server.py                         # local hybrid mode
    GEMINI_API_KEY=... python app/server.py      # with Gemini generation
    python app/server.py --mode gcp              # Vertex AI backend
    python app/server.py --port 8080             # custom port
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "config"))

# Load the core query engine from app.py
_spec = importlib.util.spec_from_file_location("app_core", ROOT / "app" / "app.py")
_mod  = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(_mod)                           # type: ignore[union-attr]

query_fn       = _mod.query
compute_diff   = _mod.compute_diff
validate_resp  = _mod.validate_response
load_chunks    = _mod.load_chunks if hasattr(_mod, "load_chunks") else _mod._load_chunks
CHUNKS_DIR     = _mod.CHUNKS_DIR

# Try to load .env for GEMINI_API_KEY
try:
    from config_env import load_env
    load_env()
except Exception:
    pass

# Pre-load chunk index at startup so first query is fast
_mod._chunks_cache = load_chunks(CHUNKS_DIR)

app = Flask(__name__, template_folder="templates", static_folder="static")
_DEFAULT_MODE = "local"


@app.route("/")
def index():
    has_key  = bool(os.environ.get("GEMINI_API_KEY"))
    n_chunks = len(_mod._chunks_cache or [])
    mode     = app.config.get("QUERY_MODE", _DEFAULT_MODE)
    return render_template("index.html",
                           has_key=has_key, n_chunks=n_chunks, mode=mode)


@app.route("/api/query", methods=["POST"])
def api_query():
    data        = request.get_json(force=True) or {}
    question    = (data.get("question") or "").strip()
    old_version = data.get("old_version") or None
    new_version = data.get("new_version") or None
    top_k       = int(data.get("top_k") or 8)
    mode        = app.config.get("QUERY_MODE", _DEFAULT_MODE)

    if not question:
        return jsonify({"error": "question is required"}), 400

    t0   = time.time()
    resp = query_fn(question, mode=mode, k=top_k,
                    old_version=old_version, new_version=new_version)
    resp.setdefault("notes", []).append(f"Latency: {(time.time()-t0)*1000:.0f}ms")

    ok, errs = validate_resp(resp)
    resp["_schema_valid"] = ok
    resp["_schema_errors"] = errs
    return jsonify(resp)


@app.route("/api/diff", methods=["POST"])
def api_diff():
    data       = request.get_json(force=True) or {}
    policy_id  = (data.get("policy_id") or "").strip()
    old_v      = (data.get("old_version") or "").strip()
    new_v      = (data.get("new_version") or "").strip()
    section    = data.get("section") or None

    if not policy_id or not old_v or not new_v:
        return jsonify({"error": "policy_id, old_version, new_version are required"}), 400

    diffs = compute_diff(policy_id, old_v, new_v, section=section)
    return jsonify({"diffs": diffs, "count": len(diffs)})


@app.route("/api/policies")
def api_policies():
    """List available policy IDs and versions from the chunk index."""
    seen: dict = {}
    for c in (_mod._chunks_cache or []):
        pid = c.get("policy_id", "")
        ver = c.get("version_date", "")
        seen.setdefault(pid, set()).add(ver)
    result = [
        {"policy_id": pid, "versions": sorted(vers)}
        for pid, vers in sorted(seen.items())
    ]
    return jsonify(result)


def main():
    parser = argparse.ArgumentParser(description="UHC Change Intelligence — Web UI")
    parser.add_argument("--mode",  choices=["local", "gcp"], default="local")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.config["QUERY_MODE"] = args.mode
    print(f"\n  UHC Change Intelligence UI")
    print(f"  Mode     : {args.mode}")
    print(f"  Chunks   : {len(_mod._chunks_cache or [])}")
    print(f"  Gemini   : {'✓ key found' if os.environ.get('GEMINI_API_KEY') else '✗ no key (rule-based fallback)'}")
    print(f"  URL      : http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
