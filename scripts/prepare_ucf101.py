#!/usr/bin/env python
"""Extract the UCF101 exercise subset from the Kaggle mirror.

Expected zip layout (pevogam/ucf101):
    UCF101/UCF-101/<ClassName>/v_<ClassName>_gXX_cYY.avi

Output:
    data/raw/ucf101/clips/<mapped_class>/*.avi
    data/raw/ucf101/manifest.csv
    data/raw/ucf101/classes.json
    data/raw/ucf101/splits.json

The default 4 classes are the ones UCF101 actually covers:
squat, pushup, jumping_jack, lunge.  UCF101 has no sit-up.
Use --include-other to add a 5th "other" class from the remaining
exercise classes (useful as hard negatives).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from pathlib import Path

import pandas as pd

TARGET = {
    "BodyWeightSquats": "squat",
    "PushUps": "pushup",
    "JumpingJack": "jumping_jack",
    "Lunges": "lunge",
}
OTHER = ["BenchPress", "HandstandPushups", "JumpRope", "PullUps"]


def group_of(name: str) -> str:
    m = re.search(r"_g(\d+)_", name)
    return f"g{m.group(1)}" if m else "g00"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/ucf101/ucf101_kaggle.zip")
    ap.add_argument("--out", default="data/raw/ucf101")
    ap.add_argument("--include-other", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"zip not found: {zip_path}")

    mapping = dict(TARGET)
    if args.include_other:
        for c in OTHER:
            mapping[c] = "other"

    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for cls, target in mapping.items():
            members = [n for n in names if f"UCF-101/{cls}/" in n and n.lower().endswith(".avi")]
            print(f"{cls}: {len(members)} videos -> {target}")
            dst_dir = clips / target
            dst_dir.mkdir(parents=True, exist_ok=True)
            for n in members:
                fn = Path(n).name
                dst = dst_dir / fn
                if not dst.exists():
                    with zf.open(n) as src, dst.open("wb") as f:
                        f.write(src.read())
                rows.append({
                    "video_path": dst.as_posix(),
                    "label": target,
                    "video_id": f"{target}_{dst.stem}",
                    "group": group_of(fn),
                    "source": "ucf101",
                })

    df = pd.DataFrame(rows)
    # Split by group, so no video/background group leaks across splits.
    groups = sorted(df["group"].unique())
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, int(round(n * 0.7)))
    n_val = max(1, int(round(n * 0.15)))
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    df["split"] = df["group"].map(lambda g: "train" if g in train_g else ("val" if g in val_g else "test"))
    df.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")

    classes = sorted(df["label"].unique())
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps({
        "train_groups": sorted(train_g),
        "val_groups": sorted(val_g),
        "test_groups": sorted(groups[n_train + n_val:]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
    print(f"manifest: {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
