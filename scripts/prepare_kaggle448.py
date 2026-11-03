#!/usr/bin/env python
"""Prepare the Kaggle 448-sequence dataset into 71-D pose sequences.

Input:
    data/raw/kaggle448/extracted/{labels,landmarks}.csv

Output:
    data/processed/kaggle448/seq/{video_id}.npz
    data/processed/kaggle448/manifest.csv
    data/processed/kaggle448/classes.json
    data/processed/kaggle448/stats.json

The source uses MediaPipe 33 keypoints; we map them to COCO-17 so the same
feature builder can be used later with YOLO-pose outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build import build_sequence  # noqa: E402

# MediaPipe-33 name -> COCO-17 index
MP_TO_COCO = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

CLASS_ORDER = ["squat", "pushup", "situp", "jumping_jack", "pullup"]
NAME_MAP = {
    "squat": "squat",
    "push_up": "pushup",
    "situp": "situp",
    "jumping_jack": "jumping_jack",
    "pull_up": "pullup",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/kaggle448/extracted")
    ap.add_argument("--out", default="data/processed/kaggle448")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    seq_dir = out / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(raw / "labels.csv")
    labels["target"] = labels["class"].map(NAME_MAP)
    if labels["target"].isna().any():
        bad = labels[labels["target"].isna()]["class"].unique().tolist()
        raise ValueError(f"unmapped classes: {bad}")

    # Read only the 33*3 coordinate columns + ids.
    lm = pd.read_csv(raw / "landmarks.csv")
    coord_cols = ["vid_id", "frame_order"]
    for name in MP_TO_COCO:
        coord_cols += [f"x_{name}", f"y_{name}"]
    lm = lm[coord_cols].sort_values(["vid_id", "frame_order"]).reset_index(drop=True)

    rows = []
    for vid_id, g in lm.groupby("vid_id", sort=True):
        label = labels.loc[labels["vid_id"] == vid_id, "target"]
        if len(label) != 1:
            continue
        g = g.sort_values("frame_order")
        xy = np.zeros((len(g), 17, 2), dtype=np.float32)
        conf = np.ones((len(g), 17), dtype=np.float32)
        for name, idx in MP_TO_COCO.items():
            xy[:, idx, 0] = g[f"x_{name}"].to_numpy(dtype=np.float32)
            xy[:, idx, 1] = g[f"y_{name}"].to_numpy(dtype=np.float32)
        feats, valid = build_sequence(xy, conf)
        dst = seq_dir / f"{int(vid_id):04d}.npz"
        np.savez_compressed(dst, features=feats, valid_ratio=np.float32(valid), fps=np.float32(30.0))
        rows.append({
            "video_id": int(vid_id),
            "label": label.iloc[0],
            "n_frames": int(len(g)),
            "seq_path": str(dst.relative_to(out)).replace("\\", "/"),
            "source": "kaggle448",
        })

    df = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)
    if len(df) != 448:
        print(f"WARNING: expected 448 sequences, got {len(df)}")

    # 70/15/15 split by video id, stratified by class.
    strat = df["label"]
    train_val, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=strat
    )
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, random_state=42, stratify=train_val["label"]
    )
    split_map = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        for vid in part["video_id"]:
            split_map[int(vid)] = name
    df["split"] = df["video_id"].map(split_map)
    df.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")

    classes = [c for c in CLASS_ORDER if c in set(df["label"])]
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "n_sequences": int(len(df)),
        "n_frames_total": int(df["n_frames"].sum()),
        "class_counts": df["label"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "note": "No subject id in source; split is by video id only (weak split).",
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
