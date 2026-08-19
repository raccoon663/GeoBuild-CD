# WHU-CD fc_siam_diff Results

- 数据：WHU-CD-256（train 4,059 / val 779 / test 2,596，spatial split 见 `reports/whu_spatial_split.md`）。
- 日志：`work/opencd/fc_siam_diff_whucd_100e/20260818_152457/20260818_152457.log`

## Validation 曲线（per epoch）

| Epoch | mIoU | aAcc | mFscore | changed IoU | changed F1 | changed P | changed R |
|---|---|---|---|---|---|---|---|
| 10 | 47.29 | 94.57 | 97.21 | 0.00 | nan | nan | 0.00 |
| 20 | 56.78 | 89.92 | 66.58 | 23.96 | 38.65 | 28.86 | 58.49 |
| 30 | 49.70 | 81.74 | 60.31 | 18.45 | 31.15 | 19.58 | 76.13 |
| 40 | 60.92 | 91.57 | 71.14 | 30.60 | 46.86 | 35.61 | 68.51 |
| 50 | 58.71 | 90.22 | 68.95 | 27.59 | 43.24 | 31.56 | 68.67 |
| 60 | 62.42 | 92.13 | 72.70 | 33.03 | 49.66 | 38.04 | 71.51 |
| 70 | 63.88 | 92.64 | 74.17 | 35.43 | 52.32 | 40.35 | 74.39 |
| 80 | 71.07 | 95.45 | 80.70 | 46.88 | 63.83 | 56.15 | 73.95 |
| 90 | 67.83 | 94.25 | 77.87 | 41.66 | 58.82 | 48.11 | 75.66 |
| 100 | 64.66 | 92.99 | 74.91 | 36.62 | 53.61 | 41.82 | 74.66 |

**Best changed IoU**: epoch 80, changed IoU 46.88, F1 63.83, mIoU 71.07.

## All-unchanged collapse 排查记录

- 早期（epoch 10）可能出现 changed recall=0（all-unchanged）；随后恢复正常（见上表与 [`docs/methodology.md`](../docs/methodology.md) §5）。

Val curves: `figures/fc_siam_diff_whu_val_curves.png`

## Test metrics（官方 test split）

```json
{
  "model": "FC-Siam-diff (best epoch 80)",
  "dataset": "WHU-CD-256 (spatial split; test 2596 tiles)",
  "checkpoint": "work/opencd/fc_siam_diff_whucd_100e\\best_mIoU_epoch_80.pth",
  "test": {
    "changed": {
      "F1": 70.77,
      "Precision": 59.72,
      "Recall": 86.84,
      "IoU": 54.77
    },
    "unchanged": {
      "F1": 98.62,
      "Precision": 99.49,
      "Recall": 97.76,
      "IoU": 97.27
    },
    "aAcc": 97.36,
    "mFscore": 84.7,
    "mIoU": 76.02
  },
  "best_val": {
    "epoch": 80,
    "changed_IoU": 46.88,
    "changed_F1": 63.83,
    "mIoU": 71.07
  }
}
```
