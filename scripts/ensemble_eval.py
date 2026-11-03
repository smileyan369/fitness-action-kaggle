#!/usr/bin/env python
"""Ensemble the best combined checkpoints and report window/clip metrics.

Example:
    python scripts/ensemble_eval.py --data data/processed/combined \
        --runs runs --out runs/ensemble_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import WindowDataset
from src.models.temporal import build_model


def find_checkpoints(runs: Path, top_k: int) -> list[Path]:
    scored = []
    for m in runs.glob("combined_*/metrics.json"):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            scored.append((float(d.get("test_macro_f1", 0)), m.parent / "best.pt"))
        except Exception:
            pass
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k] if p.exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/combined")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/ensemble_combined")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4, help="same eval stride for all models")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "manifest.csv")
    part = df[df.split == args.split].reset_index(drop=True)
    if part.empty:
        raise SystemExit(f"empty split {args.split}")
    classes = json.loads((data_dir / "classes.json").read_text(encoding="utf-8"))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ckpts = find_checkpoints(Path(args.runs), args.top_k)
    if not ckpts:
        raise SystemExit("no combined checkpoints found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_probs = []
    meta = None
    for ckpt_path in ckpts:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ck["config"]
        ds = WindowDataset(
            part, data_dir, class_to_idx,
            seq_len=int(cfg.get("seq_len", 48)),
            stride=args.stride,
            mean=ck["mean"], std=ck["std"],
        )
        ds.return_meta = True
        loader = DataLoader(
            ds, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        model = build_model(
            ck.get("arch", cfg.get("arch", "bilstm")),
            in_dim=71, n_action=len(classes), n_phase=0,
            hidden=int(cfg.get("hidden", 96)), dropout=float(cfg.get("dropout", 0.2)),
        )
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        probs = []
        cur_meta = []
        with torch.no_grad():
            for x, y, vid, start in loader:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(x.to(device))["action"]
                probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                cur_meta.extend(zip(vid, y.tolist(), start.tolist()))
        probs = np.concatenate(probs, axis=0)
        if meta is None:
            meta = cur_meta
        elif len(cur_meta) != len(meta):
            raise SystemExit("window count mismatch between checkpoints")
        all_probs.append(probs)
        print(f"{ckpt_path}: windows={len(probs)} arch={ck.get('arch', cfg.get('arch'))}")

    probs = np.mean(all_probs, axis=0)
    y_true = np.array([m[1] for m in meta])
    y_pred = probs.argmax(1)
    window_acc = float((y_true == y_pred).mean())
    window_f1 = float(f1_score(y_true, y_pred, average="macro"))

    # clip-level: average window probabilities per video_id
    by_video: dict[str, list[np.ndarray]] = {}
    labels: dict[str, int] = {}
    for p, (vid, y, _) in zip(probs, meta):
        by_video.setdefault(str(vid), []).append(p)
        labels[str(vid)] = int(y)
    clip_true, clip_pred = [], []
    for vid, ps in by_video.items():
        clip_true.append(labels[vid])
        clip_pred.append(int(np.mean(ps, axis=0).argmax()))
    clip_acc = float(np.mean(np.asarray(clip_true) == np.asarray(clip_pred)))
    clip_f1 = float(f1_score(clip_true, clip_pred, average="macro"))

    cm = confusion_matrix(clip_true, clip_pred, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix_clip.csv", encoding="utf-8-sig")
    metrics = {
        "checkpoints": [str(p) for p in ckpts],
        "n_windows": int(len(y_true)),
        "n_clips": int(len(by_video)),
        "window_accuracy": window_acc,
        "window_macro_f1": window_f1,
        "clip_accuracy": clip_acc,
        "clip_macro_f1": clip_f1,
        "per_class_clip": classification_report(
            clip_true, clip_pred, target_names=classes, output_dict=True, zero_division=0
        ),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
