#!/usr/bin/env python
"""Evaluate a trained checkpoint on a manifest split.

Examples:
    python -m src.evaluate --checkpoint runs/mmfit_bilstm/best.pt \
        --data data/processed/mmfit --split test --out runs/mmfit_bilstm/eval_test

    # MM-Fit unseen sessions 09/10/11 (stored in manifest_all.csv)
    python -m src.evaluate --checkpoint runs/mmfit_bilstm/best.pt \
        --data data/processed/mmfit --manifest manifest_all.csv --split unseen \
        --out runs/mmfit_bilstm/eval_unseen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import WindowDataset
from src.models.temporal import build_model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        out = model(x.to(device))["action"]
        y_true.extend(y.tolist())
        y_pred.extend(out.argmax(1).cpu().tolist())
    return y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    cfg = ckpt["config"]
    data_dir = Path(args.data)
    df = pd.read_csv(data_dir / args.manifest)
    part = df[df.split == args.split].reset_index(drop=True)
    if part.empty:
        raise SystemExit(f"no rows for split={args.split}")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ds = WindowDataset(
        part, data_dir, class_to_idx,
        seq_len=int(cfg.get("seq_len", 48)),
        stride=int(cfg.get("stride_eval", 4)),
        mean=ckpt["mean"], std=ckpt["std"],
    )
    workers = int(cfg.get("workers", 0))
    loader = DataLoader(
        ds, batch_size=int(cfg.get("batch_size", 32)), shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model = build_model(
        ckpt.get("arch", cfg.get("arch", "bilstm")),
        in_dim=71, n_action=len(classes), n_phase=0,
        hidden=int(cfg.get("hidden", 96)), dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    y_true, y_pred = evaluate(model, loader, device)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv", encoding="utf-8-sig")
    metrics = {
        "split": args.split,
        "n_windows": len(ds),
        "accuracy": acc,
        "macro_f1": f1,
        "per_class": classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

