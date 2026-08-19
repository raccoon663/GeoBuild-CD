# Sample Outputs — Schema Documentation

Actual candidate data and review cards are derived from restricted local
imagery and are **not** redistributed. This folder documents the output schemas
so the pipeline output format is reproducible without the data.

## `outputs/review_candidates/*.csv` / `*.gpkg`

One row per change candidate object (extracted from the ChangeFormer change
probability at threshold 0.5, then connected-component labelling):

| Column | Description |
| ------ | ----------- |
| `candidate_id` | stable object id (e.g. `CF_004503`) |
| `area_m2` | object area in m² |
| `size_group` | one of `tiny_<10m2`, `10-30m2`, `30-100m2`, `100-500m2`, `>500m2` |
| `x`, `y` | representative coordinate (lon/lat) |
| `changeformer_mean_prob` / `changeformer_max_prob` | object-level probability stats |
| `legacy_iter6000_overlap`, `legacy_iter14000_overlap`, `fc_siam_overlap` | agreement features (fraction of object covered by each other model's change mask) |
| `boundary_distance_px` | min distance to the Jan∩Apr content boundary (boundary-artifact risk) |
| `distance_to_building_m`, `jan_building_overlap`, `apr_building_overlap` | building-context features |
| `mean_spectral_diff`, `building_edge_fraction`, `compactness` | risk features |
| `building_context_score`, `agreement_score`, `boundary_artifact_risk`, `registration_risk`, `seasonal_landcover_risk`, `size_score`, `rank_score`, `candidate_rank` | transparent ranking components — see [`configs/candidate_ranking.yaml`](../configs/candidate_ranking.yaml) |

## `outputs/human_review/review_manifest.csv` (schema)

Review spreadsheet with human-review columns at the front (values after review):

| Column | Values |
| ------ | ------ |
| `review_status` | `pending` / `reviewed` / `recheck` |
| `true_change` | `yes` / `no` / `uncertain` |
| `change_type` | `new_building` / `demolition` / `extension` / `roof_or_structure_change` / `non_building_change` / `uncertain` |
| `confidence` | `high` / `medium` / `low` |
| `notes` | free text |

`outputs/human_review/review_manifest.xlsx` adds hyperlinks to the review cards
(`./cards/<candidate_id>.png`), dropdowns, and conditional formatting
(`scripts/build_review_manifest_xlsx.py`).

## `outputs/human_review/cards/<candidate_id>.png`

Three-panel review cards (Jan / Apr / overlay with candidate boundary) cropped
from the candidate geometry bbox + 150 m context buffer. QA thresholds in
`configs/candidate_ranking.yaml` → `card.qa`.

## Post-review analysis

After humans fill in the manifest columns, `scripts/after_human_review.py`
computes Precision@50/100/200, and — if the reviewed GT coverage is sufficient —
Recall@K / F1 / IoU, false-positive taxonomy, and small-change recall.
