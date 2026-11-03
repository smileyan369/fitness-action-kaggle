#!/usr/bin/env python
"""Extract selected MM-Fit pose/label files from the S3 ZIP without downloading 1.7 GB.

The S3 object is a normal ZIP; we read its central directory with HTTP Range
requests and fetch only the requested members.

Usage:
    python scripts/extract_mmfit_poses.py --only w00
    python scripts/extract_mmfit_poses.py --sessions 01,02,03,04,06,07,08,16,17,18,14,15,19,00,05,12,13,20
"""
from __future__ import annotations

import argparse
import struct
import time
import zlib
from pathlib import Path

import requests

URL = "https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip"
CHUNK = 2 * 1024 * 1024
EOCD = b"PK\x05\x06"
CD = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RangeReader:
    def __init__(self, url: str):
        self.url = url
        self.s = requests.Session()
        head = self.s.head(url, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        log(f"remote size={self.size/1e9:.3f} GB")

    def read(self, start: int, end: int, retries: int = 8) -> bytes:
        """Inclusive byte range."""
        if start < 0:
            start = self.size + start
        if end < 0:
            end = self.size + end
        end = min(end, self.size - 1)
        last = None
        for attempt in range(1, retries + 1):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 180))
                if r.status_code == 206:
                    return r.content
                if r.status_code == 200:
                    # Server ignored Range; only safe for small tail/central reads.
                    return r.content[start:end + 1]
                last = f"HTTP {r.status_code}"
            except Exception as e:
                last = str(e)
            log(f"range {start}-{end} attempt {attempt}/{retries}: {last}")
            time.sleep(min(20, attempt * 2))
        raise RuntimeError(f"range read failed: {last}")


def find_eocd(rr: RangeReader) -> tuple[int, int, int]:
    for tail in (1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024):
        data = rr.read(max(0, rr.size - tail), rr.size - 1)
        pos = data.rfind(EOCD)
        if pos >= 0:
            eocd_abs = rr.size - len(data) + pos
            (_, disk, cd_disk, n_disk, n_total, cd_size, cd_off, comment_len) = struct.unpack_from(
                "<4sHHHHIIH", data, pos
            )
            if n_total == 0xFFFF or cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
                raise RuntimeError("ZIP64 not supported by this small extractor")
            return cd_off, cd_size, n_total
    raise RuntimeError("EOCD not found")


def parse_central(rr: RangeReader, cd_off: int, cd_size: int) -> list[dict]:
    data = rr.read(cd_off, cd_off + cd_size - 1)
    entries = []
    p = 0
    while p + 46 <= len(data) and data[p:p + 4] == CD:
        fields = struct.unpack_from("<4sHHHHHHIIIHHHHHII", data, p)
        (sig, ver_made, ver_need, flags, method, mtime, mdate, crc, comp, uncomp,
         name_len, extra_len, comment_len, disk_start, int_attr, ext_attr, local_off) = fields
        name = data[p + 46:p + 46 + name_len].decode("utf-8", errors="replace")
        entries.append({
            "name": name, "flags": flags, "method": method, "crc": crc,
            "comp": comp, "uncomp": uncomp, "local_off": local_off,
        })
        p += 46 + name_len + extra_len + comment_len
    return entries


def extract_member(rr: RangeReader, ent: dict) -> bytes:
    # Local header: 30 bytes + name + extra. Use local header's name/extra lengths.
    head = rr.read(ent["local_off"], ent["local_off"] + 30 + 4096)
    if head[:4] != LOCAL:
        raise RuntimeError(f"bad local header for {ent['name']}")
    (_, ver, flags, method, mtime, mdate, crc, comp, uncomp, name_len, extra_len) = struct.unpack_from(
        "<4sHHHHHIIIHH", head, 0
    )
    data_start = ent["local_off"] + 30 + name_len + extra_len
    comp_size = ent["comp"] if ent["comp"] else comp
    if comp_size == 0:
        # Data descriptor / empty; fetch a bit more and use central dir size if available.
        comp_size = ent["comp"]
    raw = rr.read(data_start, data_start + comp_size - 1)
    if ent["method"] == 0:
        out = raw
    elif ent["method"] == 8:
        out = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f"unsupported compression method {ent['method']}")
    if len(out) != ent["uncomp"]:
        raise RuntimeError(f"size mismatch {ent['name']}: {len(out)} != {ent['uncomp']}")
    if ent["crc"] and (zlib.crc32(out) & 0xFFFFFFFF) != ent["crc"]:
        raise RuntimeError(f"crc mismatch {ent['name']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/mmfit/annotations")
    ap.add_argument("--sessions", default="", help="comma-separated two-digit session ids")
    ap.add_argument("--only", default="", help="single session id, e.g. w00")
    ap.add_argument("--poses", action="store_true", help="also fetch pose_2d.npy (large)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rr = RangeReader(URL)
    cd_off, cd_size, n_total = find_eocd(rr)
    log(f"central directory: {n_total} entries, {cd_size} bytes")
    entries = parse_central(rr, cd_off, cd_size)
    log(f"parsed {len(entries)} entries")

    wanted_sessions: set[str] = set()
    if args.only:
        wanted_sessions = {args.only.replace("w", "")}
    elif args.sessions:
        wanted_sessions = {x.strip().replace("w", "") for x in args.sessions.split(",") if x.strip()}
    else:
        wanted_sessions = {f"{i:02d}" for i in range(21)}

    targets = []
    for ent in entries:
        name = ent["name"]
        if not name.startswith("mm-fit/") or not name.endswith((".csv", ".npy")):
            continue
        parts = name.split("/")
        if len(parts) < 3:
            continue
        w = parts[1].replace("w", "")
        if w not in wanted_sessions:
            continue
        if name.endswith("_labels.csv") or (args.poses and name.endswith("_pose_2d.npy")):
            targets.append(ent)
    targets.sort(key=lambda e: e["name"])
    log(f"targets={len(targets)}")
    total_bytes = sum(e["comp"] for e in targets)
    log(f"compressed bytes to fetch={total_bytes/1e6:.1f} MB")

    for i, ent in enumerate(targets, 1):
        name = ent["name"]
        dst = out / Path(name).name
        if dst.exists() and dst.stat().st_size == ent["uncomp"]:
            log(f"SKIP {dst.name}")
            continue
        log(f"[{i}/{len(targets)}] {name} ({ent['comp']/1e6:.1f} MB compressed)")
        data = extract_member(rr, ent)
        dst.write_bytes(data)
        log(f"DONE {dst.name} ({len(data)/1e6:.1f} MB)")
    log("ALL DONE")


if __name__ == "__main__":
    main()
