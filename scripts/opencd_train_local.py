"""Phase 9 - exploratory OpenCD training on local weak labels.

Runs FC-Siam-diff on one of the locally prepared weak-label pairs
(data/processed/opencd_local/<pair>).  Results are EXPLORATORY ONLY: the
labels are derived from Jan GT XOR SegFormer pseudo-building and contain model
errors, so no real-world accuracy claim is made.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENCD_DIR = PROJECT_ROOT / "third_party" / "open-cd"
VENV_PY = r"<OPENCD_VENV_PYTHON>"


def main():
    pair = os.environ.get("LOCAL_PAIR", "2026-02")
    max_iters = int(os.environ.get("OPENCD_MAX_ITERS", "1200"))
    data_root = PROJECT_ROOT / "data" / "processed" / "opencd_local" / pair
    cfg_path = OPENCD_DIR / "configs" / "fcsn" / "fc_siam_diff_local_weak.py"
    text = (
        OPENCD_DIR / "configs" / "fcsn" / "fc_siam_diff_256x256_40k_levircd.py"
    ).read_text(encoding="utf-8")
    override = f"""
# ---- local weak-label experiment (exploratory, Phase 9) ----
data_root = r'{data_root}'
train_dataloader = dict(
    batch_size=4, num_workers=0, persistent_workers=False,
    dataset=dict(data_root=r'{data_root}'))
val_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    dataset=dict(data_root=r'{data_root}'))
test_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    dataset=dict(data_root=r'{data_root}'))
train_cfg = dict(type='IterBasedTrainLoop', max_iters={max_iters}, val_interval={max_iters // 2})
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval={max_iters // 2}, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, img_shape=(1024, 1024, 3)))
"""
    cfg_path.write_text(text + "\n" + override, encoding="utf-8")
    work_dir = PROJECT_ROOT / "work" / "opencd" / f"fc_siam_diff_local_{pair}"
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PY), str(OPENCD_DIR / "tools" / "train.py"),
        str(cfg_path), "--work-dir", str(work_dir),
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(OPENCD_DIR))
    if r.returncode != 0:
        print("train failed", r.returncode)
        return 1
    bests = sorted(work_dir.glob("best_*.pth"))
    if bests:
        cmd2 = [
            str(VENV_PY), str(OPENCD_DIR / "tools" / "test.py"),
            str(cfg_path), str(bests[-1]), "--work-dir", str(work_dir),
        ]
        print("running test:", " ".join(cmd2))
        subprocess.run(cmd2, cwd=str(OPENCD_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
