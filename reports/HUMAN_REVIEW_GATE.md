# HUMAN REVIEW GATE（Phase Y）

- 日期：2026-08-19。主模型：ChangeFormer（WHU-CD 训练，Shanghai Jan-Apr zero-shot，threshold 0.5）。
- **Total ChangeFormer candidates: 5579**

## Candidate counts by size

- tiny_<10m2: 3450
- 10-30m2: 800
- 30-100m2: 600
- 100-500m2: 481
- >500m2: 248

## Candidate counts by max probability

- max prob 0.5-0.7: 3095
- max prob 0.7-0.8: 740
- max prob 0.8-0.9: 630
- max prob >=0.9: 1114

## Top-K composition

| K | candidates |
|---|---|
| Top 50 | 50 |
| Top 100 | 100 |
| Top 200 | 200 |

## Model agreement (ChangeFormer vs others)

- ChangeFormer ∩ Original App (iter-14000): 15.2% of CF candidates
- ChangeFormer ∩ Legacy local (iter-6000): 8.8%
- ChangeFormer ∩ FC-Siam: 31.9%
- Original App-only objects（CF 未覆盖）: 9373
- FC-Siam-only objects（CF/iter-14000 未覆盖）: 29808

## Risk / small-change

- Boundary-risk candidates (boundary_distance<=16 px): 2750
- Small-change candidates (10-100 m2): 1400（10-30 与 30-100 m² 为重点审核范围，未删除）

## Review-card QA（2026-08-19 修复后）

- 卡片按候选几何 bbox+150 m 缓冲裁剪，三面板叠加候选边界；QA 阈值见 [`configs/candidate_ranking.yaml`](../configs/candidate_ranking.yaml) 的 `card.qa` 与 [`docs/methodology.md`](../docs/methodology.md) §7。
- 本次 295 张卡片 QA：invalid/flag 0 张（要求 0，已满足）。

## 建议人工审查规模

- Review pool（Top200 ∪ stratified）: 295 个对象；分层审查集 166 个（含 15% 负控）。
- 建议首轮人工审查 **Top 100**（约 1-2 小时）；完整 review pool 约 295 张卡片。

## 路径

- Review cards: `outputs/human_review/cards/`
- review_manifest.csv: `outputs/human_review/review_manifest.csv`
- stratified_review_set.csv: `outputs/human_review/stratified_review_set.csv`
- 全部候选目录: `outputs/review_candidates/`（all_candidates / top50/100/200 / oa_only / fc_only / negative_controls，gpkg+csv）

## 下一步（人工 review 完成后执行）

在 `outputs/human_review/review_manifest.csv` 中填写最后 5 列（review_status / true_change / change_type / confidence / notes）后，运行：

```
python scripts/after_human_review.py
```

将计算 Precision@50/100/200；若人工 GT 覆盖充分再计算 Recall@K / F1 / IoU、false-positive taxonomy、small-change recall 与模型 agreement 效用，并决定阈值校准/微调策略。

**当前 STOP：等待人工 review。不计算 Shanghai accuracy，不把 candidate 当作 confirmed change。**
