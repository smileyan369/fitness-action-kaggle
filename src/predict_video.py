#!/usr/bin/env python
"""Single-video inference: YOLO-pose -> 71-D features -> temporal checkpoint.

Example:
    python -m src.predict_video --checkpoint ../cloud_results/runs/combined_tcn_s42/best.pt \
        --video my_squat.mp4 --model yolov8n-pose.pt --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.features.build import build_sequence, make_windows
from src.models.temporal import build_model


def extract_keypoints(video_path: str, model: YOLO, conf: float = 0.3, imgsz: int = 640):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    xy_list, conf_list = [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            xy_list.append(np.zeros((17, 2), np.float32))
            conf_list.append(np.zeros((17,), np.float32))
        else:
            boxes = r.boxes.xyxy.detach().cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int(np.argmax(areas))
            xy_list.append(r.keypoints.xy[i].detach().cpu().numpy().astype(np.float32))
            conf_list.append(r.keypoints.conf[i].detach().cpu().numpy().astype(np.float32))
        frame_idx += 1
    cap.release()
    return np.stack(xy_list), np.stack(conf_list), fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ck["classes"]
    cfg = ck["config"]
    seq_len = int(cfg.get("seq_len", 48))
    stride = int(cfg.get("stride_eval", 4))
    model = build_model(
        ck.get("arch", cfg.get("arch", "bilstm")),
        in_dim=71, n_action=len(classes), n_phase=0,
        hidden=int(cfg.get("hidden", 96)), dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    pose = YOLO(args.model)
    xy, conf, fps = extract_keypoints(args.video, pose, args.conf, args.imgsz)
    feats, valid = build_sequence(xy, conf)
    windows, starts = make_windows(feats, seq_len=seq_len, stride=stride)
    if len(windows) == 0:
        raise SystemExit("no pose windows")
    windows = (windows - ck["mean"]) / ck["std"]

    probs = []
    with torch.no_grad():
        for i in range(0, len(windows), 64):
            x = torch.from_numpy(windows[i:i + 64].astype(np.float32)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)["action"]
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    mean_prob = probs.mean(axis=0)
    top = int(mean_prob.argmax())

    result = {
        "video": str(args.video),
        "fps": fps,
        "n_frames": int(len(feats)),
        "valid_ratio": valid,
        "prediction": classes[top],
        "confidence": float(mean_prob[top]),
        "mean_probabilities": {c: float(p) for c, p in zip(classes, mean_prob)},
        "windows": [
            {"start_frame": int(s), "prediction": classes[int(p.argmax())], "confidence": float(p.max())}
            for s, p in zip(starts, probs)
        ],
    }
    out = Path(args.out) if args.out else Path(args.video).with_suffix(".prediction.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("prediction", "confidence", "n_frames", "valid_ratio")},
                     ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
