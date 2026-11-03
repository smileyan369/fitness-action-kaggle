#!/usr/bin/env python
"""Extract 17 COCO keypoints from videos with YOLO-pose.

Manifest columns (required):
    video_path,label,video_id
Optional:
    group,start_frame,end_frame

Example:
    python -m src.pose.extract \
        --manifest data/raw/ucf101/manifest.csv \
        --out data/processed/ucf101 \
        --model yolov8n-pose.pt --device cpu --batch 16
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise SystemExit(f"ultralytics is required: {e}")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def extract_one(
    model: YOLO,
    video_path: str,
    start_frame: int | None,
    end_frame: int | None,
    stride: int,
    batch_size: int,
    conf: float,
    imgsz: int,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = max(0, int(start_frame or 0))
    end = total if end_frame is None else min(total, int(end_frame))
    if end <= start:
        end = total
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    xy_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    idx = start

    def flush():
        nonlocal frames
        if not frames:
            return
        results = model.predict(
            frames, conf=conf, imgsz=imgsz, verbose=False, device=device,
            half=bool(device and str(device).startswith("cuda")),
        )
        for r in results:
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                xy_all.append(np.zeros((17, 2), dtype=np.float32))
                conf_all.append(np.zeros((17,), dtype=np.float32))
                continue
            boxes = r.boxes.xyxy.detach().cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int(np.argmax(areas))
            k = r.keypoints.xy[i].detach().cpu().numpy().astype(np.float32)
            c = r.keypoints.conf[i].detach().cpu().numpy().astype(np.float32)
            if k.shape[0] != 17:
                k = np.zeros((17, 2), dtype=np.float32)
                c = np.zeros((17,), dtype=np.float32)
            xy_all.append(k)
            conf_all.append(c)
        frames = []

    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % stride == 0:
            frames.append(frame)
            if len(frames) >= batch_size:
                flush()
        idx += 1
    flush()
    cap.release()

    if not xy_all:
        xy = np.zeros((0, 17, 2), dtype=np.float32)
        cf = np.zeros((0, 17), dtype=np.float32)
    else:
        xy = np.stack(xy_all, axis=0)
        cf = np.stack(conf_all, axis=0)
    return xy, cf, fps / max(1, stride), idx - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    kp_dir = out / "keypoints"
    kp_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    required = {"video_path", "label", "video_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")

    if args.device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"model={args.model} device={device} videos={len(manifest)} batch={args.batch}")
    model = YOLO(args.model)

    rows = []
    limit = args.limit if args.limit > 0 else len(manifest)
    for n, row in enumerate(tqdm(manifest.itertuples(index=False), total=min(limit, len(manifest)))):
        if n >= limit:
            break
        video_id = safe_name(getattr(row, "video_id"))
        start = int(getattr(row, "start_frame")) if hasattr(row, "start_frame") and not pd.isna(getattr(row, "start_frame")) else None
        end = int(getattr(row, "end_frame")) if hasattr(row, "end_frame") and not pd.isna(getattr(row, "end_frame")) else None
        try:
            xy, cf, fps, n_read = extract_one(
                model, str(row.video_path), start, end, max(1, args.stride),
                args.batch, args.conf, args.imgsz, device,
            )
        except Exception as e:
            print(f"[skip] {row.video_path}: {e}")
            continue
        if len(xy) == 0:
            print(f"[skip] no frames: {row.video_path}")
            continue
        dst = kp_dir / f"{video_id}.npz"
        np.savez_compressed(dst, xy=xy, conf=cf, fps=np.float32(fps), n_read=np.int32(n_read))
        rows.append({
            "video_id": video_id,
            "label": getattr(row, "label"),
            "group": getattr(row, "group", ""),
            "seq_path": str(dst.relative_to(out)).replace("\\", "/"),
            "n_frames": int(len(xy)),
            "fps": float(fps),
            "split": getattr(row, "split", ""),
            "source": getattr(row, "source", ""),
        })
    pd.DataFrame(rows).to_csv(out / "keypoints_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"done: {len(rows)} sequences -> {out}")


if __name__ == "__main__":
    main()
