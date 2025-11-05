# Create an ingest.py script tailored to your Vertex Workbench + GCS setup.
# It:
# - reads docs/manifest.csv
# - downloads each PDF to a temp file from gcs_uri
# - computes sha256 if missing and updates manifest.csv
# - parses PDF into {policy_id, version_date, section, page, text}
# - chunks text to ~800-1000 "tokens" with ~15% overlap (using tiktoken if available; fallback to words)
# - writes data/parsed/*.jsonl and data/chunks/*.jsonl

import os, json, csv, re, hashlib, tempfile, sys, io
from pathlib import Path
from typing import List, Dict, Tuple
import tiktoken
from pypdf import PdfReader
from pdfminer.high_level import extract_text
import tempfile

BASE = "../"

DOCS_CSV = os.path.join(BASE, "docs", "manifest.csv")
PARSED_DIR = os.path.join(BASE, "data", "parsed")
CHUNKS_DIR = os.path.join(BASE, "data", "chunks")
os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

def _gcs_to_temp(gcs_uri: str) -> str:
    from google.cloud import storage
    parts = gcs_uri.replace("gs://","").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    blob.download_to_filename(tmp)
    return tmp

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_pages_pypdf(local_pdf: str):
    reader = PdfReader(local_pdf)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i+1, txt))
    return pages

def extract_pages_pdfminer(local_pdf: str):
    text = extract_text(local_pdf) or ""
    # crude split by form feed or page indicators if present
    pages = []
    chunks = re.split(r'\f+', text) if '\f' in text else [text]
    for i, chunk in enumerate(chunks):
        pages.append((i+1, chunk))
    return pages

import re

# --- tuning knobs ---
MAX_HEADINGS_PER_PAGE = 12   # hard cap to avoid over-splitting
KNOWN_HEADINGS = {
    "coverage rationale": "Coverage Rationale",
    "benefit considerations": "Benefit Considerations",
    "definitions": "Definitions",
    "applicable codes": "Applicable Codes",
    "description of services": "Description of Services",
    "clinical evidence": "Clinical Evidence",
    "references": "References",
    "policy history": "Policy History",
    "instructions for use": "Instructions for Use",
    "background": "Background",
    "centers for medicare and medicaid services": "CMS",
    "u.s. food and drug administration": "FDA",
    "table of contents": "Table of Contents",
}

HEADER_PATTERNS = [
    r"^unitedhealthcare.*(medical benefit|medical policy).*$",
    r"^proprietary information of unitedhealthcare.*$",
    r"^page \d+ of \d+.*$",
    r"^table of contents.*$",  # we will skip TOC section later
]

MIN_TOKENS = 120  # ~100–150 tokens

BULLETS = {"•", "", "●", "·", "–", "-", "—", "–", "•", ""}  # includes the odd glyph

def canonicalize(title: str) -> str:
    t = title.strip()
    k = t.lower()
    for key, val in KNOWN_HEADINGS.items():
        if key in k:
            return val
    return t

def clean_line(line: str) -> str:
    s = line.strip()
    # drop common header/footer lines
    for pat in HEADER_PATTERNS:
        if re.match(pat, s, flags=re.I):
            return ""
    # normalize weird bullets/symbols and soft hyphens
    s = s.replace("\u00ad", "")              # soft hyphen
    s = s.replace("", "- ").replace("•", "- ").replace("●", "- ").replace("·", "- ").replace("", "Note: ")
    # collapse multiple spaces
    s = re.sub(r"\s{2,}", " ", s)
    return s

def is_title_case_like(s: str) -> bool:
    words = s.split()
    if not words or len(words) > 10:
        return False
    def ok(w):
        # allow FDA/CMS etc. as uppercase & normal Title Case for others
        return w.isupper() or (w[:1].isupper() and w[1:].islower())
    return sum(1 for w in words if ok(w)) / len(words) >= 0.8

def is_heading(line: str, next_line: str | None) -> bool:
    s = line.strip()
    if not s or len(s) < 6 or len(s) > 80:
        return False
    # avoid sentences / table captions
    if s.endswith((".", ":", ";", ",")):
        return False
    # avoid bullet lines unless explicitly known
    if s[:1] in BULLETS and s.upper() not in (k.upper() for k in KNOWN_HEADINGS.values()):
        return False
    # exact known headings (case-insensitive)
    if s.lower() in KNOWN_HEADINGS:
        return True
    # ALL-CAPS heuristic (letters only)
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        cap_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if cap_ratio > 0.85 and any(ch.isalpha() for ch in s):
            return True
    # Title-Case heuristic, require a blank line after to reduce FPs
    has_blank_after = (next_line is not None and next_line.strip() == "")
    return is_title_case_like(s) and has_blank_after

from datetime import datetime

EFF_PATTERNS = [
    r"\bEffective\s+Date[:\s]+(\d{1,2}/\d{1,2}/\d{4})\b",
    r"\bEffective\s+(\d{1,2}/\d{1,2}/\d{4})\b",
    r"\bEffective\s+Date[:\s]+([A-Za-z]+ \d{1,2}, \d{4})\b",
    r"\bEffective\s+([A-Za-z]+ \d{1,2}, \d{4})\b",
]

MONTHS = {m: i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}

def _norm_eff_date(s: str) -> str:
    s = s.strip()
    # try M/D/YYYY
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm, dd, yy = map(int, m.groups())
        return f"{yy:04d}-{mm:02d}-{dd:02d}"
    # try Month D, YYYY
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$", s)
    if m:
        mm = MONTHS[m.group(1).capitalize()]
        dd = int(m.group(2)); yy = int(m.group(3))
        return f"{yy:04d}-{mm:02d}-{dd:02d}"
    return s  # fallback (leave as-is)

def extract_effective_from_text(text: str) -> str | None:
    for pat in EFF_PATTERNS:
        m = re.search(pat, text, flags=re.I)
        if m:
            return _norm_eff_date(m.group(1))
    return None

def parse_pdf_sections(local_pdf: str):
    # Try pypdf then fallback to pdfminer
    try:
        pages = extract_pages_pypdf(local_pdf)
    except Exception:
        try:
            pages = extract_pages_pdfminer(local_pdf)
        except Exception as e:
            print(f"[warn] Failed parsing {local_pdf}: {e}")
            return []

    # Optional: capture "Effective" date (first page), return via section records
    effective_from = None
    if pages:
        first_raw = pages[0][1] or ""
        effective_from = extract_effective_from_text(first_raw)

    sects = []
    current = {"section": "Introduction", "page": 1, "text": ""}
    prev_page = None

    for page_num, raw_text in pages:
        if not raw_text:
            continue
        # reset per-page heading cap
        page_new_sections = 0

        # clean lines & drop empties
        lines = [clean_line(l) for l in raw_text.splitlines()]
        lines = [l for l in lines if l]

        for idx, line in enumerate(lines):
            nxt = lines[idx + 1] if idx + 1 < len(lines) else None
            if is_heading(line, nxt) and page_new_sections < MAX_HEADINGS_PER_PAGE:
                # push previous section if it has content
                if current["text"].strip():
                    sects.append(current)
                current = {
                    "section": canonicalize(line[:120]),
                    "page": page_num,
                    "text": ""
                }
                page_new_sections += 1
            else:
                current["text"] += line + "\n"

        # page break
        current["text"] += "\n"
        prev_page = page_num

    if current["text"].strip():
        sects.append(current)

    # Merge tiny sections into the previous one
    merged = []
    for sec in sects:
        tok_len = tokenize_len(sec["text"])
        if merged and tok_len < MIN_TOKENS:
            merged[-1]["text"] += "\n" + sec["text"]
        else:
            merged.append(sec)
    sects = merged

    # Fallback if nothing detected
    if not sects:
        for page_num, text in pages:
            if text and text.strip():
                sects.append({"section": f"Page {page_num}", "page": page_num, "text": text})

    # Optionally drop the Table of Contents section entirely
    sects = [s for s in sects if s["section"].lower() != "table of contents"]

    # Attach effective_from for downstream (if you want it now)
    for s in sects:
        s["effective_from"] = effective_from

    return sects

def _enc():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

ENC = _enc()

def tokenize_len(s: str) -> int:
    if ENC:
        return len(ENC.encode(s))
    words = re.findall(r"\w+", s)
    return max(1, int(len(words)/0.75))

def smart_chunks(text: str, target_tokens: int = 900, overlap_ratio: float = 0.15):
    if not text:
        return []
    if ENC:
        toks = ENC.encode(text)
        n = len(toks)
        step = max(1, int(target_tokens * (1 - overlap_ratio)))
        chunks = []
        for start in range(0, n, step):
            end = min(n, start + target_tokens)
            if start >= end:
                break
            piece = ENC.decode(toks[start:end])
            chunks.append({"text": piece, "start_tok": start, "end_tok": end})
            if end == n:
                break
        return chunks
    else:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        def para_tokens(p): return tokenize_len(p)
        chunks, cur, cur_tok = [], [], 0
        for p in paras:
            pt = para_tokens(p)
            if cur_tok + pt <= target_tokens or not cur:
                cur.append(p); cur_tok += pt
            else:
                chunks.append("\n\n".join(cur))
                cur = [p]; cur_tok = pt
        if cur:
            chunks.append("\n\n".join(cur))
        out, cursor = [], 0
        for c in chunks:
            tlen = tokenize_len(c)
            out.append({"text": c, "start_tok": cursor, "end_tok": cursor + tlen})
            cursor += max(1, int(target_tokens * (1 - overlap_ratio)))
        return out
    
def process_row(row):
    gcs_uri = row.get("gcs_uri")
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        raise ValueError("Expected gcs_uri (gs://...) for ingestion")
    local_pdf = _gcs_to_temp(gcs_uri)
    try:
        file_sha = sha256_file(local_pdf)
        sections = parse_pdf_sections(local_pdf)
        policy_id = row.get("policy_id","")
        version_date = row.get("version_date","").strip()
        # quick: make it filesystem-safe (turn 10/1/2025 into 10-1-2025)
        version_date = version_date.replace("/", "-")
        # write parsed
        parsed_path = os.path.join(PARSED_DIR, f"{policy_id}_{version_date}.jsonl")
        with open(parsed_path, "w", encoding="utf-8") as w:
            for s in sections:
                w.write(json.dumps({
                    "policy_id": policy_id,
                    "version_date": version_date,
                    "section": s["section"],
                    "page": s["page"],
                    "text": s["text"],
                    "effective_from": s.get("effective_from"),
                }, ensure_ascii=False) + "\n")
        # chunks
        chunk_recs = []
        for s_idx, s in enumerate(sections):
            chs = smart_chunks(s["text"], target_tokens=900, overlap_ratio=0.15)
            for i, ch in enumerate(chs):
                chunk_recs.append({
                    "policy_id": policy_id,
                    "version_date": version_date,
                    "section": s["section"],
                    "page": s["page"],
                    "chunk_id": f"{policy_id}:{version_date}:S{s_idx}:P{s['page']}:C{i}",
                    "start_tok": ch["start_tok"],
                    "end_tok": ch["end_tok"],
                    "text": ch["text"],
                    "effective_from": s.get("effective_from"),
                })
        chunks_path = os.path.join(CHUNKS_DIR, f"{policy_id}_{version_date}.jsonl")
        with open(chunks_path, "w", encoding="utf-8") as w:
            for c in chunk_recs:
                w.write(json.dumps(c, ensure_ascii=False) + "\n")
        return len(sections), len(chunk_recs), file_sha
    finally:
        if os.path.exists(local_pdf):
            os.remove(local_pdf)
            
def main():
    # load manifest
    with open(DOCS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No rows in manifest")
        return
    required = {"payer","lob","policy_id","version_date","gcs_uri"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"manifest.csv missing columns: {missing}")
    updated = False
    for row in rows:
        try:
            sec_n, ch_n, sha = process_row(row)
            if (not row.get("sha256")) and sha:
                row["sha256"] = sha; updated = True
            print(f"OK {row.get('policy_id')} {row.get('version_date')}: {sec_n} sections, {ch_n} chunks")
        except Exception as e:
            print(f"ERR {row.get('policy_id')} {row.get('version_date')}: {e}")
    if updated:
        # write back manifest with sha256 updates
        with open(DOCS_CSV, "w", newline="", encoding="utf-8") as w:
            wr = csv.DictWriter(w, fieldnames=rows[0].keys())
            wr.writeheader()
            wr.writerows(rows)
        print("manifest.csv updated (sha256).")
 

if __name__ == "__main__":
    main()