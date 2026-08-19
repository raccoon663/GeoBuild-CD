"""OpenCD public benchmark sanity test (Phase 8).

Downloads/extracts LEVIR-CD (if not present), writes a reduced-iteration
FC-Siam-diff config, trains briefly, and runs validation.

Requires the separate OpenCD venv: <OPENCD_VENV_PYTHON>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENCD_DIR = PROJECT_ROOT / "third_party" / "open-cd"
DATA_DIR = OPENCD_DIR / "data" / "LEVIR-CD"
TARBALL = PROJECT_ROOT / "third_party" / "datasets" / "LEVIR-CD.tar.gz"
VENV_PY = r"<OPENCD_VENV_PYTHON>"


def ensure_dataset():
    if DATA_DIR.exists():
        n = len(list((DATA_DIR / "train" / "A").glob("*.png"))) if (DATA_DIR / "train" / "A").exists() else 0
        if n:
            print("LEVIR-CD present:", n, "train A tiles")
            return
    if not TARBALL.exists():
        raise SystemExit("LEVIR-CD tarball not found; download first")
    print("extracting", TARBALL)
    with tarfile.open(TARBALL) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        tf.extractall(DATA_DIR.parent, members=members)
    # the archive may contain a nested LEVIR-CD folder
    if not (DATA_DIR / "train").exists():
        cands = [p for p in DATA_DIR.parent.iterdir() if p.is_dir() and "levir" in p.name.lower()]
        for c in cands:
            if (c / "train").exists():
                if DATA_DIR.exists():
                    shutil.rmtree(DATA_DIR)
                shutil.move(str(c), str(DATA_DIR))
                break
    n = len(list((DATA_DIR / "train" / "A").glob("*.png"))) if (DATA_DIR / "train" / "A").exists() else 0
    print("extracted, train A tiles:", n)


def write_smoke_config() -> Path:
    max_iters = int(os.environ.get("OPENCD_MAX_ITERS", "800"))
    val_interval = max(200, max_iters // 2)
    src = OPENCD_DIR / "configs" / "fcsn" / "fc_siam_diff_256x256_40k_levircd.py"
    text = src.read_text(encoding="utf-8")
    # must live inside the open-cd configs tree so `_base_` resolves
    work_cfg = OPENCD_DIR / "configs" / "fcsn" / "fc_siam_diff_smoke_levircd.py"
    work_cfg.parent.mkdir(parents=True, exist_ok=True)
    override = f"""
# ---- smoke-test overrides (Phase 8 sanity) ----
data_root = r'{str(DATA_DIR)}'
train_dataloader = dict(batch_size=4, num_workers=0, persistent_workers=False)
val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)
test_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)
train_cfg = dict(type='IterBasedTrainLoop', max_iters={max_iters}, val_interval={val_interval})
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=400, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, img_shape=(1024, 1024, 3)))
"""
    # The base configs set data_root/dataloaders; appending overrides works via Config merge at load.
    work_cfg.write_text(text + "\n" + override, encoding="utf-8")
    return work_cfg


def main():
    ensure_dataset()
    cfg = write_smoke_config()
    work_dir = PROJECT_ROOT / "work" / "opencd" / "fc_siam_diff_smoke"
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PY), str(OPENCD_DIR / "tools" / "train.py"),
        str(cfg), "--work-dir", str(work_dir),
    ]
    print("running:", " ".join(cmd))
    env = os.environ.copy()
    r = subprocess.run(cmd, cwd=str(OPENCD_DIR), env=env)
    print("train exit", r.returncode)
    best = work_dir / "best_mIoU_iter_800.pth"
    if not best.exists():
        bests = sorted(work_dir.glob("best_*.pth"))
        best = bests[0] if bests else None
    if best is not None:
        cmd2 = [
            str(VENV_PY), str(OPENCD_DIR / "tools" / "test.py"),
            str(cfg), str(best), "--work-dir", str(work_dir),
        ]
        print("running test:", " ".join(cmd2))
        subprocess.run(cmd2, cwd=str(OPENCD_DIR), env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
