# HUMAN REVIEW GATE (Phase Y)

**Archived prototype status:** This gate records the implemented candidate-review
workflow. The project was intentionally paused before systematic human
validation; the 295-card pool is not used to claim Shanghai accuracy.

- Date: 2026-08-19. Primary model: ChangeFormer (WHU-CD trained, Shanghai
  Jan–Apr zero-shot, threshold 0.5).
- **Total ChangeFormer candidates: 5,579** (candidate count does not imply
  correctness).

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
- Original App-only objects (not covered by CF): 9,373
- FC-Siam-only objects (not covered by CF / iter-14000): 29,808

## Risk / small-change

- Boundary-risk candidates (boundary_distance <= 16 px): 2,750
- Small-change candidates (10-100 m²): 1,400 (the 10-30 and 30-100 m² ranges
  are the priority review range and are kept, not deleted)

## Review-card QA (after the 2026-08-19 fix)

- Cards are cropped from the candidate geometry bbox + 150 m context buffer
  with a three-panel layout and the candidate boundary overlaid; QA thresholds
  are in [`configs/candidate_ranking.yaml`](../configs/candidate_ranking.yaml)
  (`card.qa`) and [`docs/methodology.md`](../docs/methodology.md) §7.
- 295-card QA: 0 invalid / flagged (requirement 0 — satisfied).

## Review pool size

- Review pool (Top-200 ∪ stratified): 295 objects; stratified review set 166
  (incl. 15% negative controls).
- Suggested first pass: **Top 100** (~1–2 hours); full pool ~295 cards.

## Outputs

- Review cards: `outputs/human_review/cards/`
- review_manifest.csv: `outputs/human_review/review_manifest.csv`
- stratified_review_set.csv: `outputs/human_review/stratified_review_set.csv`
- Candidate catalogs: `outputs/review_candidates/` (all_candidates /
  top50/100/200 / oa_only / fc_only / negative_controls; gpkg + csv)

## After human review (deferred)

A future evaluation script would compute Precision@K after verified
annotations become available; this phase is not included in the current frozen
prototype. Given sufficient reviewed GT coverage it would also compute
Recall@K / F1 / IoU, a false-positive taxonomy, small-change recall, and model
agreement utility, and would inform threshold calibration / fine-tuning.

**STOP status:** systematic human review and Shanghai accuracy computation are
intentionally deferred; candidates are not treated as confirmed change.
