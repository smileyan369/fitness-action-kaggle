#!/usr/bin/env python
"""Prepare MM-Fit 2D pose sequences for the 5-class model.

MM-Fit stores OpenPose COCO-18 2D poses at video frame indices:
    pose_2d shape = (2, N, 19): [x/y, frame, 18 joints]
    labels.csv    = start_frame, end_frame, reps, activity

We map COCO-18 -> COCO-17 (drop neck), build 71-D features for every
exercise set, and use the official session split.

Download the needed files first:
    python scripts/extract_mmfit_poses.py --poses \
        --sessions 01,02,03,04,06,07,08,16,17,18,14,15,19,00,05,12,13,20
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

ACTIVITY_MAP = {
    "squats": "squat",
    "pushups": "pushup",
    "situps": "situp",
    "jumping_jacks": "jumping_jack",
    "lunges": "lunge",
}
CLASS_ORDER = ["squat", "pushup", "situp", "jumping_jack", "lunge"]

TRAIN = {"01", "02", "03", "04", "06", "07", "08", "16", "17", "18"}
VAL = {"14", "15", "19"}
TEST = {"00", "05", "12", "13", "20"}
UNSEEN = {"09", "10", "11"}

# OpenPose COCO-18 joint -> column in pose_2d (column 0 is frame index).
# 0 nose, 1 neck, 2 Rshoulder, 3 Relbow, 4 Rwrist, 5 Lshoulder, 6 Lelbow,
# 7 Lwrist, 8 Rhip, 9 Rknee, 10 Rankle, 11 Lhip, 12 Lknee, 13 Lankle,
# 14 Reye, 15 Leye, 16 Rear, 17 Lear
OPENPOSE_TO_COCO17 = {
    "nose": 1,
    "left_eye": 16,
    "right_eye": 15,
    "left_ear": 18,
    "right_ear": 17,
    "left_shoulder": 6,
    "right_shoulder": 3,
    "left_elbow": 7,
    "right_elbow": 4,
    "left_wrist": 8,
    "right_wrist": 5,
    "left_hip": 12,
    "right_hip": 9,
    "left_knee": 13,
    "right_knee": 10,
    "left_ankle": 14,
    "right_ankle": 11,
}


def split_of(session: str) -> str:
    if session in TRAIN:
        return "train"
    if session in VAL:
        return "val"
    if session in TEST:
        return "test"
    return "unseen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/mmfit/annotations")
    ap.add_argument("--out", default="data/processed/mmfit")
    ap.add_argument("--min-frames", type=int, default=10)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    seq_dir = out / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_path in sorted(raw.glob("w*_labels.csv")):
        session = label_path.stem.split("_")[0].replace("w", "")
        pose_path = raw / f"w{session}_pose_2d.npy"
        if not pose_path.exists():
            print(f"[skip] missing pose: {pose_path}")
            continue
        pose = np.load(pose_path, allow_pickle=True)
        if pose.ndim != 3 or pose.shape[0] != 2:
            print(f"[skip] bad pose shape {pose.shape}: {pose_path}")
            continue
        frames = pose[0, :, 0]
        labels = pd.read_csv(label_path, header=None, names=["start", "end", "reps", "activity"])
        for k, row in labels.iterrows():
            act = str(row["activity"]).strip()
            if act not in ACTIVITY_MAP:
                continue
            mask = (frames >= float(row["start"])) & (frames <= float(row["end"]))
            if mask.sum() < args.min_frames:
                print(f"[skip] short {session} {act} set {k}: {mask.sum()} frames")
                continue
            seg = pose[:, mask, :]  # (2, T, 19)
            xy = np.zeros((seg.shape[1], 17, 2), dtype=np.float32)
            for j, (name, col) in enumerate(OPENPOSE_TO_COCO17.items()):
                xy[:, j, 0] = seg[0, :, col]
                xy[:, j, 1] = seg[1, :, col]
            conf = np.ones((seg.shape[1], 17), dtype=np.float32)
            feats, valid = build_sequence(xy, conf)
            vid = f"{session}_{ACTIVITY_MAP[act]}_{k:02d}"
            dst = seq_dir / f"{vid}.npz"
            np.savez_compressed(
                dst, features=feats, valid_ratio=np.float32(valid),
                fps=np.float32(30.0), start=int(row["start"]), end=int(row["end"]),
            )
            rows.append({
                "video_id": vid,
                "label": ACTIVITY_MAP[act],
                "group": session,
                "split": split_of(session),
                "n_frames": int(len(feats)),
                "seq_path": str(dst.relative_to(out)).replace("\\", "/"),
                "source": "mmfit_pose2d",
                "reps": int(row["reps"]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no sequences prepared; download pose_2d files first")
    # Keep only official train/val/test for the main manifest.
    official = df[df.split.isin(["train", "val", "test"])].copy()
    official.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    df.to_csv(out / "manifest_all.csv", index=False, encoding="utf-8-sig")
    classes = [c for c in CLASS_ORDER if c in set(df.label)]
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "n_sequences": int(len(df)),
        "n_official": int(len(official)),
        "class_counts": df.label.value_counts().to_dict(),
        "split_counts": df.split.value_counts().to_dict(),
        "sessions": sorted(df.group.unique().tolist()),
        "note": "Official MM-Fit session split; each sequence is one exercise set.",
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
