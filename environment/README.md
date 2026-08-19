# Environments

GeoBuild-CD uses **two separate Python environments**:

| Environment | Purpose | Definition |
| ----------- | ------- | ---------- |
| **A — base / geospatial / legacy** | pure-PyTorch SegFormer inference, geospatial IO, candidate ranking, review-card tooling | [`../requirements.txt`](../requirements.txt) |
| **B — OpenCD training** | FC-Siam-diff / ChangeFormer training and zero-shot inference | notes below (external deps) |

Do **not** expect `pip install -r requirements.txt` to install a complete
OpenCD training stack. The two runtimes intentionally differ in PyTorch/CUDA
versions.

## Environment A — base / geospatial / legacy

Tested: Windows, Python 3.10, CUDA 12.x.

```bash
python -m pip install -r requirements.txt
```

This covers `src/segmentation/` (pure PyTorch, no mmcv/mmseg needed), all
`scripts/*.py` geospatial tooling (WHU-CD audit/prep, registration analysis,
candidate extraction, review-card generation), and the review-manifest
spreadsheet builder.

## Environment B — OpenCD training

Requires a separate virtual environment with an OpenCD-compatible stack.
The versions below are the **tested environment** recorded in the project
setup notes (Windows wheels were only available for this older torch); adjust
to your own CUDA/PyTorch stack if needed — treat this as "tested", not
"required exact".

```bash
# create the venv (example name: opencd)
python -m venv <OPENCD_VENV>

# install the CUDA build of torch used in this project
<OPENCD_VENV_PYTHON> -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# OpenMMLab stack (tested versions)
<OPENCD_VENV_PYTHON> -m pip install mmengine==0.10.4 mmcv==2.1.0 \
    mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0

# OpenCD itself (upstream project, Apache-2.0)
# clone https://github.com/likyoo/open-cd into third_party/open-cd, then:
<OPENCD_VENV_PYTHON> -m pip install -e third_party/open-cd
```

> The project configs in `configs/changeformer/` and `configs/fcsn/` are
> OpenCD run configs; they expect OpenCD cloned at `third_party/open-cd`
> (see `scripts/opencd_train_local.py` and `scripts/opencd_run_benchmark.py`).
> The WHU-CD dataset itself is downloaded separately (see
> [`../docs/data_and_splits.md`](../docs/data_and_splits.md)) and is not
> vendored.
