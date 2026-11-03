#!/usr/bin/env python
"""Merge multiple processed pose datasets into one training manifest.

Example:
    python scripts/merge_manifests.py \
        --inputs data/processed/mmfit,data/processed/ucf101 \
        --out data/processed/combined
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="comma-separated processed dirs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inputs = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    out = Path(args.out)
    seq_dir = out / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in inputs:
        manifest_path = src / "manifest.csv"
        if not manifest_path.exists():
            raise SystemExit(f"missing manifest: {manifest_path}")
        df = pd.read_csv(manifest_path)
        prefix = src.name
        for row in df.itertuples(index=False):
            seq_src = src / str(row.seq_path)
            if not seq_src.exists():
                print(f"[skip] missing seq: {seq_src}")
                continue
            vid = f"{prefix}_{row.video_id}"
            dst = seq_dir / f"{vid}.npz"
            if not dst.exists():
                shutil.copy2(seq_src, dst)
            rows.append({
                "video_id": vid,
                "label": row.label,
                "group": f"{prefix}_{getattr(row, 'group', 'na')}",
                "split": getattr(row, "split", "train"),
                "n_frames": int(getattr(row, "n_frames", 0)),
                "seq_path": str(dst.relative_to(out)).replace("\\", "/"),
                "source": getattr(row, "source", prefix) if not pd.isna(getattr(row, "source", prefix)) else prefix,
            })

    merged = pd.DataFrame(rows)
    # Drop non-official splits from the main training manifest.
    merged = merged[merged["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    merged.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    classes = sorted(merged["label"].unique().tolist())
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "n_sequences": int(len(merged)),
        "class_counts": merged["label"].value_counts().to_dict(),
        "split_counts": merged["split"].value_counts().to_dict(),
        "sources": merged["source"].value_counts().to_dict(),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
