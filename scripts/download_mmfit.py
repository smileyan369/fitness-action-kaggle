#!/usr/bin/env python
"""Download MM-Fit RGB videos from Zenodo with resume + MD5 verification.

Usage:
    python scripts/download_mmfit.py --out data/raw/mmfit --workers 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api/records/7607736"
CHUNK = 8 * 1024 * 1024
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(item: dict, out_dir: Path, retries: int = 8) -> Path:
    key = item["key"]
    size = int(item["size"])
    expected_md5 = item.get("checksum", "").replace("md5:", "")
    url = item["links"]["self"]
    final = out_dir / key
    part = out_dir / (key + ".part")

    if final.exists() and final.stat().st_size == size:
        if expected_md5 and md5sum(final) == expected_md5:
            log(f"SKIP {key} (already complete)")
            return final
        log(f"WARN {key} size ok but md5 mismatch, re-download")
        final.unlink()

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                if have and r.status_code == 206:
                    mode = "ab"
                    log(f"RESUME {key} at {have/1e9:.2f} GB")
                elif have and r.status_code == 200:
                    mode = "wb"
                    have = 0
                    log(f"RESTART {key} (server ignored Range)")
                elif r.status_code in (200, 206):
                    mode = "wb"
                    log(f"START {key} ({size/1e9:.2f} GB)")
                else:
                    raise RuntimeError(f"HTTP {r.status_code}")
                total = have
                last = time.time()
                with part.open(mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        total += len(chunk)
                        now = time.time()
                        if now - last > 20:
                            pct = 100 * total / size if size else 0
                            log(f"  {key}: {total/1e9:.2f}/{size/1e9:.2f} GB ({pct:.1f}%)")
                            last = now
            if part.stat().st_size != size:
                raise RuntimeError(f"size mismatch: {part.stat().st_size} != {size}")
            if expected_md5:
                got = md5sum(part)
                if got != expected_md5:
                    raise RuntimeError(f"md5 mismatch: {got} != {expected_md5}")
            os.replace(part, final)
            log(f"DONE {key}")
            return final
        except Exception as e:
            log(f"ERROR {key} attempt {attempt}/{retries}: {e}")
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"failed after {retries} attempts: {key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/mmfit")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated file keys, e.g. w00_rgb.mp4,w01_rgb.mp4")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = requests.get(ZENODO_API, timeout=60).json()
    (out_dir / "zenodo_record.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    items = [f for f in meta["files"] if f["key"].endswith("_rgb.mp4")]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        items = [f for f in items if f["key"] in wanted]
    items.sort(key=lambda x: x["key"])
    total = sum(int(f["size"]) for f in items)
    log(f"files={len(items)} total={total/1e9:.2f} GB -> {out_dir.resolve()}")

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(download_one, f, out_dir): f["key"] for f in items}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                fut.result()
            except Exception as e:
                log(f"FAILED {key}: {e}")
                failures.append(key)
    if failures:
        log(f"FAILURES: {failures}")
        sys.exit(1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
