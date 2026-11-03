#!/usr/bin/env python
"""Three-slot training queue for a single GPU.

Starts up to --slots training processes at once.  A job with a missing
`requires` path stays pending, so MM-Fit jobs can run while UCF101 is
being extracted, and combined jobs start as soon as combined data exists.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG_DIR = ROOT / "runs"
STATUS = LOG_DIR / "queue_status.json"

JOBS = [
    # MM-Fit baseline / variants
    {"name": "mmfit_bilstm_s42", "data": "data/processed/mmfit",
     "config": "configs/bilstm_mmfit.yaml", "seed": 42, "out": "runs/mmfit_bilstm_s42"},
    {"name": "mmfit_bilstm_h128_s2024", "data": "data/processed/mmfit",
     "config": "configs/bilstm_mmfit_h128.yaml", "seed": 2024, "out": "runs/mmfit_bilstm_h128_s2024"},
    {"name": "mmfit_tcn_s42", "data": "data/processed/mmfit",
     "config": "configs/tcn_mmfit.yaml", "seed": 42, "out": "runs/mmfit_tcn_s42"},
    {"name": "mmfit_bilstm_s3407", "data": "data/processed/mmfit",
     "config": "configs/bilstm_mmfit.yaml", "seed": 3407, "out": "runs/mmfit_bilstm_s3407"},
    # Combined: MM-Fit + UCF101 (starts after merge)
    {"name": "combined_bilstm_s42", "data": "data/processed/combined",
     "config": "configs/bilstm_combined.yaml", "seed": 42, "out": "runs/combined_bilstm_s42",
     "requires": "data/processed/combined/manifest.csv"},
    {"name": "combined_bilstm_h128_s2024", "data": "data/processed/combined",
     "config": "configs/bilstm_combined_h128.yaml", "seed": 2024, "out": "runs/combined_bilstm_h128_s2024",
     "requires": "data/processed/combined/manifest.csv"},
    {"name": "combined_tcn_s42", "data": "data/processed/combined",
     "config": "configs/tcn_combined.yaml", "seed": 42, "out": "runs/combined_tcn_s42",
     "requires": "data/processed/combined/manifest.csv"},
    {"name": "combined_bilstm_h128_s3407", "data": "data/processed/combined",
     "config": "configs/bilstm_combined_h128.yaml", "seed": 3407, "out": "runs/combined_bilstm_h128_s3407",
     "requires": "data/processed/combined/manifest.csv"},
    {"name": "combined_bilstm_s2024", "data": "data/processed/combined",
     "config": "configs/bilstm_combined.yaml", "seed": 2024, "out": "runs/combined_bilstm_s2024",
     "requires": "data/processed/combined/manifest.csv"},
]


def load_status():
    if STATUS.exists():
        return json.loads(STATUS.read_text())
    return {"jobs": {}, "started": time.time()}


def save_status(st):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    slots = int(os.environ.get("QUEUE_SLOTS", "3"))
    st = load_status()
    running: dict[str, subprocess.Popen] = {}
    pending = [j for j in JOBS if (ROOT / j["out"] / "metrics.json").exists() is False]
    done_names = {j["name"] for j in JOBS if (ROOT / j["out"] / "metrics.json").exists()}
    for j in JOBS:
        if j["name"] in done_names:
            st["jobs"].setdefault(j["name"], {})["status"] = "done"
    save_status(st)
    print(f"queue: {len(pending)} pending, {len(done_names)} already done, slots={slots}", flush=True)

    while True:
        for name, p in list(running.items()):
            code = p.poll()
            if code is not None:
                running.pop(name)
                st["jobs"][name]["status"] = "done" if code == 0 else "failed"
                st["jobs"][name]["exit_code"] = code
                st["jobs"][name]["finished"] = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] {name} exit={code}", flush=True)
                save_status(st)
        # launch
        for j in pending[:]:
            if len(running) >= slots:
                break
            name = j["name"]
            if name in running:
                continue
            req = j.get("requires")
            if req and not (ROOT / req).exists():
                continue
            out = ROOT / j["out"]
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / "train.log", "a", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "3",
                "MKL_NUM_THREADS": "3",
                "PYTHONUNBUFFERED": "1",
            })
            cmd = [
                PY, "-m", "src.train",
                "--data", str(ROOT / j["data"]),
                "--config", str(ROOT / j["config"]),
                "--out", str(out),
                "--seed", str(j["seed"]),
                "--device", "cuda",
            ]
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[name] = p
            pending.remove(j)
            st["jobs"][name] = {
                "status": "running", "pid": p.pid, "started": time.time(),
                "data": j["data"], "config": j["config"], "seed": j["seed"],
            }
            print(f"[{time.strftime('%H:%M:%S')}] start {name} pid={p.pid}", flush=True)
            save_status(st)

        if not running and not pending:
            print("queue empty: all jobs processed", flush=True)
            break
        if not running and pending:
            # Nothing runnable now; wait for external data prep.
            print(f"waiting: {[j['name'] for j in pending]}", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
