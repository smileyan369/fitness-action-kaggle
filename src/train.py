#!/usr/bin/env python
"""Train the pose-sequence classifier.

Example:
    python -m src.train --data data/processed/kaggle448 \
        --config configs/bilstm_kaggle448.yaml --out runs/kaggle448_bilstm
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch.utils.data import DataLoader

from src.data.dataset import WindowDataset, compute_scaler
from src.models.temporal import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              amp: bool = False, scaler=None):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []
    n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)["action"]
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total_loss += float(loss.item()) * len(y)
            n += len(y)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(out.detach().argmax(1).cpu().tolist())
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if n else 0.0
    f1 = float(f1_score(y_true, y_pred, average="macro")) if n else 0.0
    return total_loss / max(n, 1), acc, f1, y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/kaggle448")
    ap.add_argument("--config", default="configs/bilstm_kaggle448.yaml")
    ap.add_argument("--out", default="runs/kaggle448_bilstm")
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
    ap.add_argument("--seed", type=int, default=0, help="override config seed (0 = keep config)")
    ap.add_argument("--arch", default="", help="override config arch: bilstm or tcn")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.seed:
        cfg["seed"] = args.seed
    if args.arch:
        cfg["arch"] = args.arch
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    classes = json.loads((data_dir / "classes.json").read_text(encoding="utf-8"))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"classes={classes}")
    print(f"device={device} train={sum(manifest.split == 'train')} val={sum(manifest.split == 'val')} test={sum(manifest.split == 'test')}")

    seq_len = int(cfg.get("seq_len", 48))
    train_ds = WindowDataset(
        manifest[manifest.split == "train"], data_dir, class_to_idx,
        seq_len=seq_len, stride=int(cfg.get("stride_train", 8)), augment=True,
    )
    mean, std = compute_scaler(train_ds)
    train_ds.mean, train_ds.std = mean, std
    val_ds = WindowDataset(
        manifest[manifest.split == "val"], data_dir, class_to_idx,
        seq_len=seq_len, stride=int(cfg.get("stride_eval", 4)), mean=mean, std=std,
    )
    test_ds = WindowDataset(
        manifest[manifest.split == "test"], data_dir, class_to_idx,
        seq_len=seq_len, stride=int(cfg.get("stride_eval", 4)), mean=mean, std=std,
    )
    print(f"windows train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("not enough windows; check data/split")

    batch = int(cfg.get("batch_size", 32))
    workers = int(cfg.get("workers", 0))
    pin = device.type == "cuda"
    loader_kwargs = dict(
        batch_size=batch, num_workers=workers, pin_memory=pin,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    arch = str(cfg.get("arch", "bilstm"))
    model = build_model(
        arch,
        in_dim=71,
        n_action=len(classes),
        n_phase=0,
        hidden=int(cfg.get("hidden", 96)),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)
    print(f"arch={arch} params={model.n_params/1e6:.3f}M workers={workers}")

    counts = manifest[manifest.split == "train"]["label"].value_counts()
    weights = np.array([len(manifest[manifest.split == "train"]) / counts[c] for c in classes], dtype=np.float32)
    weights = weights / weights.mean()
    print("class weights:", dict(zip(classes, weights.round(3).tolist())))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=device),
        label_smoothing=float(cfg.get("label_smoothing", 0.05)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 60))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    patience = int(cfg.get("patience", 10))
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    print(f"amp={amp}")

    best_f1 = -1.0
    best_epoch = -1
    bad = 0
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, True, amp=amp, scaler=scaler
        )
        va_loss, va_acc, va_f1, y_true, y_pred = run_epoch(
            model, val_loader, criterion, optimizer, device, False, amp=amp
        )
        scheduler.step()
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(f"epoch {epoch:03d} train loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f}")
        if va_f1 > best_f1 + 1e-5:
            best_f1, best_epoch, bad = va_f1, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "classes": classes,
                "mean": mean,
                "std": std,
                "epoch": epoch,
                "val_f1": va_f1,
                "arch": arch,
                "feature_version": "v1_71",
            }, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (best={best_epoch}, f1={best_f1:.4f})")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc, te_f1, y_true, y_pred = run_epoch(
        model, test_loader, criterion, optimizer, device, False, amp=amp
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        "classes": classes,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_macro_f1": te_f1,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_test_windows": len(test_ds),
        "params_m": model.n_params / 1e6,
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "per_class": report,
        "dataset": data_dir.name,
        "note": "Split is read from manifest.csv; do not compare across datasets with different splits.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TEST acc={te_acc:.4f} macro_f1={te_f1:.4f}")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

