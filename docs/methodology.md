# Methodology

This document consolidates the supporting analysis of the GeoBuild-CD project
(the per-phase working notes that produced the seven curated reports in
[`reports/`](../reports/)).

## 1. Problem framing

- **Data stage:** same-year multi-month high-resolution imagery of a rural town in
  Shanghai (months 2026-01, 02, 03, 04, 05-31).
- **Supervision:** the **April** image carries the main precise building labels
  (5,386 footprints, used to train the original SegFormer). The **January** image
  carries an independent manual building annotation (17,292 footprints), treated
  as independent temporal evaluation data — not as a training reference month.
- **Change definition:** same-year differences are *model-derived change
  candidates*, never claimed as verified construction or demolition until a clean
  Jan–Apr change GT exists (the "GT Gate" below).

## 2. Data audit

- Imagery was audited for extent, CRS, pixel size, nodata handling and border
  artifacts; content masks exclude nodata borders from all area fractions.
- Label QA checked polygon validity, duplicates, and raster/label alignment.
- Key numbers (content-normalized):
  - Building fraction per month: 4.3% / 4.6% / 4.9% / 4.5% / 4.7%.
  - Temporal stability (Jan–May): stable 3.8%, unstable 3.0%, persistent
    non-building 93.3%; candidate gain 0.7%, loss 0.5%.
- Jan–Apr registration audit estimated global residual offsets of about
  1.6–2.3 px (ECC / phase-correlation), motivating the mask-shift sensitivity
  study in [`reports/registration_sensitivity.md`](../reports/registration_sensitivity.md).

## 3. Legacy segmentation baseline

- SegFormer-B5 implemented in pure PyTorch (`src/segmentation/`) to avoid the
  Windows mmcv/mmseg install; ImageNet normalization embedded in the model.
- **Training chain (all April-supervised):** B2 buildings → B2 hard finetune →
  B5 hard finetune → **B5 shadow finetune (iter-6000, final local model)** →
  B5 WHU-generalized finetune (**iter-14000**, shipped in the packaged app).
- Local model evaluation on the independent January labels: pixel IoU 0.485,
  pixel F1 0.653 (precision 0.606 / recall 0.707); object-level matching is
  strict (precision 0.228 / recall 0.078) because the model under-segments.
- Legacy April clean-eval (54 curated patches): building IoU 0.749 — this is an
  April-supervision number from the predecessor project, not a temporal-transfer
  result.
- **ONNX caveat:** all ONNX files in the old project are byte-identical and
  correspond to the **WHU-generalized iter-14000** weights, not the iter-6000
  local model. See [`model_provenance.md`](model_provenance.md).

## 4. Registration sensitivity (why differencing failed)

- Simulated residual shifts 0/1/2/4 px applied at image level (SegFormer
  re-inference on three 4096×4096 test regions) and at mask level (aligned
  April mask shifted then XORed).
- Mask-level result on the full AOI: 1 px → ~39.6 ha apparent change (170,939
  fragments), 2 px → ~77.9 ha, 4 px → ~151.3 ha.
- Interpretation: the Jan–Apr residual of ~1.6–2.3 px alone can account for
  **tens of hectares** of pseudo-change in the old differencing output.
  Conclusion: direct CD training must tolerate ±2 px rather than relying on a
  single global correction, and old differencing must never be read as measured
  change.

## 5. WHU-CD direct change detection

- Built WHU-CD-256: official mosaics tiled on a global 256 px grid; train mosaic
  split by a **4×3 spatial block partition** (val = blocks (0,3) and (2,0)); no
  random patch split. Official change labels used (XOR-consistency ~98%).
- Models: FC-Siam-diff and ChangeFormer (mit-b0), 100 epochs, both trained from
  scratch on the spatially separated train split.
- Test results on the official test mosaic — see
  [`reports/whu_fcsn_results.md`](../reports/whu_fcsn_results.md) and
  [`reports/whu_changeformer_results.md`](../reports/whu_changeformer_results.md).
- Pipeline sanity was first validated on LEVIR-CD (400-iter smoke run).

## 6. Zero-shot transfer to Shanghai

- April image resampled onto the January grid (bilinear) before inference; no
  Shanghai labels used.
- ChangeFormer predicts ~0.9–1.1% change of valid content; FC-Siam-diff ~20.2%
  (threshold 0.5). The two models behave very differently out-of-domain.
- Structural pre-GT comparison (old differencing vs direct CD):
  - ChangeFormer ≈ 68.4 ha vs filtered old baseline ≈ 87.4 ha, but overlap
    IoU ≈ 0.02 — largely disjoint change regions.
  - Boundary-edge share: direct CD 1.3–2.9% vs old C0.7 10.5% — direct CD shows
    less registration-edge artifact.
  - FC-Siam shows broad-area overprediction; unsuitable as primary screening
    model without recalibration.
- OpenCD direct CD outputs are binary change maps (no gain/loss direction);
  gain/loss splitting does not apply.

## 7. Candidate screening (human-in-the-loop)

- ChangeFormer change probability → connected components → size groups
  (tiny_<10, 10–30, 30–100, 100–500, >500 m²).
- Transparent ranking ([`configs/candidate_ranking.yaml`](../configs/candidate_ranking.yaml)):
  weighted combination of mean/max probability, building-context score, model
  agreement (legacy-6000 / legacy-14000 / FC-Siam), and size; penalties for
  boundary-artifact, registration risk, and seasonal/land-cover risk.
- Review-card generation crops candidate geometry + 150 m context buffer from
  the Jan and Apr rasters; QA ensures valid (non-nodata) crops.
- **Current state (HUMAN REVIEW GATE):** 5,579 ChangeFormer candidates; review
  pool of 295 objects (Top-200 ∪ stratified, incl. 15% negative controls);
  review cards QA clean (0 invalid/flagged of 295). Human review is pending;
  Precision@K etc. are computed only after review.

## 8. GT Gate

No Shanghai accuracy metrics are computed before a clean Jan–Apr change ground
truth exists. The January and April annotations follow different polygonization
conventions and cannot be XORed into a change GT; building a small, exhaustively
reviewed benchmark is future work (see [`future_work.md`](future_work.md)).

## 9. Honest-reporting rules (enforced)

1. No accuracy claims on months without human labels.
2. Pseudo-labels are never used as an independent validation set.
3. Same-year prediction differences are called **candidate change**, not
   construction/demolition.
4. No pixel-wise temporal differencing before registration is checked.
5. Legacy SegFormer work is reused, not rewritten for its own sake.
