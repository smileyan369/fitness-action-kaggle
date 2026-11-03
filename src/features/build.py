"""Build 71-D pose features from 17 COCO keypoints.

Feature layout (71):
  - 17 * 3 = 51 : normalized x, y, confidence
  - 12 angles  : 8 joint angles + torso lean + 2 body lines + shoulder tilt
  - 8 distances: shoulder width, hip width, torso length,
                 L/R hip-knee, L/R knee-ankle, mean wrist-shoulder

The same function must be used at training and inference time.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# COCO-17 index order, as used by Ultralytics YOLO-pose.
NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
IDX = {name: i for i, name in enumerate(NAMES)}

# Pairs used by the 12 angle features.
ANGLE_PAIRS = [
    ("left_elbow", ("left_shoulder", "left_elbow", "left_wrist")),
    ("right_elbow", ("right_shoulder", "right_elbow", "right_wrist")),
    ("left_shoulder", ("left_hip", "left_shoulder", "left_elbow")),
    ("right_shoulder", ("right_hip", "right_shoulder", "right_elbow")),
    ("left_hip", ("left_shoulder", "left_hip", "left_knee")),
    ("right_hip", ("right_shoulder", "right_hip", "right_knee")),
    ("left_knee", ("left_hip", "left_knee", "left_ankle")),
    ("right_knee", ("right_hip", "right_knee", "right_ankle")),
    ("left_body_line", ("left_shoulder", "left_hip", "left_ankle")),
    ("right_body_line", ("right_shoulder", "right_hip", "right_ankle")),
]

FEATURE_DIM = 17 * 3 + 12 + 8


def feature_names() -> list[str]:
    names: list[str] = []
    for n in NAMES:
        names += [f"{n}_x", f"{n}_y", f"{n}_c"]
    names += [f"angle_{n}" for n, _ in ANGLE_PAIRS]
    names += ["torso_lean", "shoulder_tilt"]
    names += [
        "dist_shoulder_width", "dist_hip_width", "dist_torso_length",
        "dist_left_hip_knee", "dist_right_hip_knee",
        "dist_left_knee_ankle", "dist_right_knee_ankle",
        "dist_wrist_shoulder_mean",
    ]
    assert len(names) == FEATURE_DIM, (len(names), FEATURE_DIM)
    return names


def _interp_missing(xy: np.ndarray, conf: np.ndarray, conf_th: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate missing keypoint coordinates along time.

    Returns:
        xy_i: (T,17,2) interpolated coordinates
        valid: (T,17) boolean mask of observed keypoints
    """
    xy = np.asarray(xy, dtype=np.float32).copy()
    conf = np.asarray(conf, dtype=np.float32)
    t = xy.shape[0]
    valid = (conf >= conf_th) & np.isfinite(xy).all(axis=-1)
    # Treat all-zero detections as missing.
    valid &= ~(np.abs(xy).sum(axis=-1) < 1e-6)
    idx = np.arange(t)
    for j in range(xy.shape[1]):
        ok = valid[:, j]
        if ok.sum() == 0:
            xy[:, j] = 0.0
            continue
        if ok.sum() == t:
            continue
        for k in range(2):
            xy[~ok, j, k] = np.interp(idx[~ok], idx[ok], xy[ok, j, k])
    return xy, valid


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    cos = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def build_sequence(xy: np.ndarray, conf: np.ndarray, conf_th: float = 0.2) -> tuple[np.ndarray, float]:
    """Convert (T,17,2)/(T,17) keypoints into (T,71) features.

    Returns:
        features: (T,71) float32
        valid_ratio: mean fraction of visible keypoints
    """
    xy, valid = _interp_missing(xy, conf, conf_th)
    t = xy.shape[0]
    if t == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), 0.0

    # Per-frame hip center; per-sequence scale from median shoulder width.
    hip_center = (xy[:, IDX["left_hip"]] + xy[:, IDX["right_hip"]]) / 2.0
    shoulder_width = np.linalg.norm(
        xy[:, IDX["left_shoulder"]] - xy[:, IDX["right_shoulder"]], axis=-1
    )
    good_sw = shoulder_width[shoulder_width > 1e-4]
    scale = float(np.median(good_sw)) if good_sw.size else 1.0
    scale = max(scale, 1e-4)

    coords = (xy - hip_center[:, None, :]) / scale
    feats = np.zeros((t, FEATURE_DIM), dtype=np.float32)
    feats[:, :51] = np.concatenate([coords, conf[..., None]], axis=-1).reshape(t, 51)

    for i in range(t):
        p = xy[i]
        for k, (_, (a, b, c)) in enumerate(ANGLE_PAIRS):
            feats[i, 51 + k] = _angle(p[IDX[a]], p[IDX[b]], p[IDX[c]])

        sh_mid = (p[IDX["left_shoulder"]] + p[IDX["right_shoulder"]]) / 2.0
        hip_mid = (p[IDX["left_hip"]] + p[IDX["right_hip"]]) / 2.0
        torso = sh_mid - hip_mid
        feats[i, 61] = np.degrees(np.arctan2(abs(torso[0]), abs(torso[1]))) if _norm(torso) > 1e-8 else 0.0
        shoulder_vec = p[IDX["right_shoulder"]] - p[IDX["left_shoulder"]]
        feats[i, 62] = np.degrees(np.arctan2(abs(shoulder_vec[1]), abs(shoulder_vec[0]))) if _norm(shoulder_vec) > 1e-8 else 0.0

        sh_w = _norm(p[IDX["left_shoulder"]] - p[IDX["right_shoulder"]])
        hip_w = _norm(p[IDX["left_hip"]] - p[IDX["right_hip"]])
        torso_len = _norm(sh_mid - hip_mid)
        l_hip_knee = _norm(p[IDX["left_hip"]] - p[IDX["left_knee"]])
        r_hip_knee = _norm(p[IDX["right_hip"]] - p[IDX["right_knee"]])
        l_knee_ankle = _norm(p[IDX["left_knee"]] - p[IDX["left_ankle"]])
        r_knee_ankle = _norm(p[IDX["right_knee"]] - p[IDX["right_ankle"]])
        wrist_sh = (
            _norm(p[IDX["left_wrist"]] - p[IDX["left_shoulder"]])
            + _norm(p[IDX["right_wrist"]] - p[IDX["right_shoulder"]])
        ) / 2.0
        feats[i, 63:71] = np.asarray(
            [sh_w, hip_w, torso_len, l_hip_knee, r_hip_knee,
             l_knee_ankle, r_knee_ankle, wrist_sh],
            dtype=np.float32,
        ) / scale

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, float(valid.mean())


def make_windows(seq: np.ndarray, seq_len: int = 48, stride: int = 8) -> tuple[np.ndarray, list[int]]:
    """Slide a fixed-length window over one sequence, padding edges.

    Returns:
        windows: (N, seq_len, F)
        starts: list of start frame indices
    """
    seq = np.asarray(seq, dtype=np.float32)
    t = len(seq)
    if t == 0:
        return np.zeros((0, seq_len, FEATURE_DIM), dtype=np.float32), []
    if t < seq_len:
        pad = np.repeat(seq[-1:], seq_len - t, axis=0)
        return np.concatenate([seq, pad], axis=0)[None], [0]
    starts = list(range(0, t - seq_len + 1, max(1, stride)))
    if starts[-1] != t - seq_len:
        starts.append(t - seq_len)
    windows = np.stack([seq[s:s + seq_len] for s in starts], axis=0)
    return windows, starts
