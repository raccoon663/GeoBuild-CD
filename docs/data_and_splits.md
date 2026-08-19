# Data and Splits

## Placeholder legend

All absolute local paths in this repository are replaced with generic
placeholders. Replace them with your own paths before running any script.

| Placeholder | Meaning |
| ----------- | ------- |
| `<LOCAL_SHANGHAI_DATA>` | root folder of the local high-resolution Shanghai imagery + labels (not redistributed) |
| `<LOCAL_LABEL_RASTER>` | ENVI label raster aligned with the April image |
| `<LOCAL_ARCGIS_PROJECT>` | ArcGIS annotation project holding the April building labels |
| `<MODEL_ROOT>` | folder containing the SegFormer / SAM checkpoints |
| `<WHU_CD_ROOT>` | local path to the prepared WHU-CD dataset (downloaded + tiled) |
| `<WHU_SEG_ROOT>` | local path to the WHU building-segmentation data (legacy finetune only) |
| `<LEGACY_PROJECT>` | local path to the predecessor segmentation project data |
| `<LEGACY_OUTPUTS>` | local path to predecessor training outputs (checkpoints/logs) |
| `<OPENCD_VENV_PYTHON>` | python executable of the separate OpenCD virtual environment |

## Shanghai imagery (not redistributed)

Same-year multi-month high-resolution imagery of a rural town in Shanghai
(~0.5 m, EPSG:4326), months 2026-01 / 02 / 03 / 04 / 05-31:

| Month | Role |
| ----- | ---- |
| 2026-01 | independent manual building labels (17,292 footprints) — temporal evaluation data |
| 2026-02 | inference only |
| 2026-03 | coarse/candidate labels only |
| 2026-04 | **main supervision** — precise building labels (5,386 footprints) used to train the SegFormer baseline |
| 2026-05-31 | inference only |

**Data-access restrictions:** the raw imagery and any full-resolution derived
products (previews, failure galleries, review cards) are **not redistributed**.
The repository ships scripts, metric plots, and schematic figures only.

**Important:** the January and April annotations follow different polygonization
conventions; they cannot be XORed into a clean Jan–Apr change GT. A clean local
change benchmark does not exist yet (GT Gate; see
[`methodology.md`](methodology.md) §8).

## WHU-CD (public benchmark, not redistributed)

Download WHU-CD from the official dataset source and configure `<WHU_CD_ROOT>`.
This repository provides only preparation code and OpenCD run configs.

### WHU-CD-256 build (`scripts/whu_cd_prepare.py`)

- Official mosaics (2012 train 21243×15354, 2012 test 11265×15354, 2016
  counterparts) tiled on a global 256 px grid.
- Train mosaic partitioned by a **4×3 spatial block grid**; **val = blocks
  (0,3) and (2,0)**; remaining blocks train; test mosaic is entirely test.
  No random patch split.
- Official change labels used (XOR-consistency with A/B ≈ 98%).

| Split | Tiles | Change fraction (pixel) |
| ----- | ----: | ----------------------: |
| train | 4059  | 4.3430%                 |
| val   | 779   | 5.4270%                 |
| test  | 2596  | 3.6830%                 |

Full audit: [`reports/whu_data_audit.md`](../reports/whu_data_audit.md) and
[`reports/whu_spatial_split.md`](../reports/whu_spatial_split.md). Split
visualization: [`figures/whu_spatial_split.png`](../figures/whu_spatial_split.png).

## Train / val / test discipline

- SegFormer local model: April labels only; January is held out as independent
  temporal evaluation.
- Direct CD models: WHU-CD spatially separated split; Shanghai is never used
  for training (zero-shot transfer only).
- No pseudo-labels are used as an independent validation set.
