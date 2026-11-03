#!/usr/bin/env python
"""Convert raw YOLO-pose keypoints (xy/conf) into 71-D feature sequences.

Input:
    data/processed/<name>/keypoints_manifest.csv
    data/processed/<name>/keypoints/<video_id>.npz  (xy, conf, fps)

Output:
    data/processed/<name>/seq/<video_id>.npz       (features, valid_ratio, fps)
    data/processed/<name>/manifest.csv
    data/processed/<name>/classes.json
    data/processed/<name>/stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build import build_sequence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    args = ap.parse_args()
    root = Path(args.processed)
    src_manifest = root / "keypoints_manifest.csv"
    if not src_manifest.exists():
        raise SystemExit(f"missing {src_manifest}")
    df = pd.read_csv(src_manifest)
    seq_dir = root / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in df.itertuples(index=False):
        raw = root / str(row.seq_path)
        if not raw.exists():
            print(f"[skip] missing {raw}")
            continue
        arr = np.load(raw)
        if "features" in arr:
            feats = arr["features"].astype(np.float32)
            valid = float(arr["valid_ratio"]) if "valid_ratio" in arr else 1.0
            fps = float(arr["fps"]) if "fps" in arr else 30.0
        else:
            feats, valid = build_sequence(arr["xy"], arr["conf"])
            fps = float(arr["fps"]) if "fps" in arr else 30.0
        dst = seq_dir / f"{row.video_id}.npz"
        np.savez_compressed(dst, features=feats, valid_ratio=np.float32(valid), fps=np.float32(fps))
        rows.append({
            "video_id": row.video_id,
            "label": row.label,
            "group": getattr(row, "group", ""),
            "split": getattr(row, "split", "train"),
            "n_frames": int(len(feats)),
            "seq_path": str(dst.relative_to(root)).replace("\\", "/"),
            "source": getattr(row, "source", ""),
        })
    out = pd.DataFrame(rows)
    out.to_csv(root / "manifest.csv", index=False, encoding="utf-8-sig")
    classes = sorted(out["label"].unique().tolist())
    (root / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "n_sequences": int(len(out)),
        "class_counts": out["label"].value_counts().to_dict(),
        "split_counts": out["split"].value_counts().to_dict(),
        "n_frames_total": int(out["n_frames"].sum()),
    }
    (root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved: {root/'manifest.csv'}")


if __name__ == "__main__":
    main()
