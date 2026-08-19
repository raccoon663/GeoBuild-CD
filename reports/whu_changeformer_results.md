# WHU-CD ChangeFormer Results

- Data: WHU-CD-256 (train 4,059 / val 779 / test 2,596; spatial split in
  [`reports/whu_spatial_split.md`](whu_spatial_split.md)).
- Log: `work/opencd/changeformer_mit-b0_whucd_100e/20260818_164749/20260818_164749.log`

## Validation curves (per epoch)

| Epoch | mIoU | aAcc | mFscore | changed IoU | changed F1 | changed P | changed R |
|---|---|---|---|---|---|---|---|
| 10 | 82.93 | 98.05 | 89.92 | 67.89 | 80.88 | 86.50 | 75.94 |
| 20 | 87.15 | 98.64 | 92.73 | 75.72 | 86.18 | 95.95 | 78.23 |
| 30 | 88.07 | 98.73 | 93.32 | 77.46 | 87.30 | 95.44 | 80.44 |
| 40 | 86.16 | 98.54 | 92.09 | 73.84 | 84.95 | 96.12 | 76.11 |
| 50 | 86.10 | 98.53 | 92.06 | 73.74 | 84.88 | 96.19 | 75.96 |
| 60 | 86.83 | 98.62 | 92.52 | 75.09 | 85.77 | 97.35 | 76.65 |
| 70 | 87.81 | 98.71 | 93.15 | 76.97 | 86.98 | 95.95 | 79.55 |
| 80 | 86.42 | 98.58 | 92.26 | 74.33 | 85.27 | 97.49 | 75.77 |
| 90 | 86.64 | 98.60 | 92.41 | 74.74 | 85.55 | 97.14 | 76.43 |
| 100 | 86.52 | 98.58 | 92.33 | 74.52 | 85.40 | 97.03 | 76.25 |

**Best changed IoU:** epoch 30, changed IoU 77.46, F1 87.30, mIoU 88.07.

## All-unchanged collapse check

- **No all-unchanged collapse observed:** from the first validation point
  (epoch 10) the changed IoU is positive (> 0); the model learned the changed
  class normally.

Val curves: `figures/changeformer_whu_val_curves.png`

## Test metrics (official test split)

```json
{
  "model": "ChangeFormer mit-b0 (best epoch 30)",
  "dataset": "WHU-CD-256 (spatial split; test 2596 tiles)",
  "checkpoint": "work/opencd/changeformer_mit-b0_whucd_100e/best_mIoU_epoch_30.pth",
  "test": {
    "changed": {
      "F1": 91.64,
      "Precision": 94.99,
      "Recall": 88.52,
      "IoU": 84.58
    },
    "unchanged": {
      "F1": 99.69,
      "Precision": 99.56,
      "Recall": 99.82,
      "IoU": 99.39
    },
    "aAcc": 99.41,
    "mFscore": 95.67,
    "mIoU": 91.98
  },
  "best_val": {
    "epoch": 30,
    "changed_IoU": 77.46,
    "changed_F1": 87.3,
    "mIoU": 88.07
  }
}
```
