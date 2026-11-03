#!/usr/bin/env python
"""Convert a keypoints_manifest.csv into the train.py manifest format.

Usage:
    python scripts/finalize_keypoints.py \
        --processed data/processed/ucf101 \
        --out data/processed/ucf101
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    p = Path(args.processed)
    out = Path(args.out) if args.out else p
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(p / "keypoints_manifest.csv")
    if "split" not in df.columns or df["split"].isna().any():
        # Fallback: split by group if available, otherwise by video_id.
        key = "group" if "group" in df.columns and df["group"].notna().any() else "video_id"
        groups = sorted(df[key].astype(str).unique())
        n = len(groups)
        train = set(groups[: max(1, int(n * 0.7))])
        val = set(groups[max(1, int(n * 0.7)): max(2, int(n * 0.85))])
        df["split"] = df[key].astype(str).map(lambda g: "train" if g in train else ("val" if g in val else "test"))
    df.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    classes = sorted(df["label"].unique().tolist())
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "n_sequences": int(len(df)),
        "class_counts": df["label"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "n_frames_total": int(df["n_frames"].sum()) if "n_frames" in df else None,
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved: {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
