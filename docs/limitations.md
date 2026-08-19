# Limitations

These limitations are part of the project's research credibility and are not
weakened by any downstream phrasing in the README.

1. **No clean local change GT.** January / April building annotations follow
   different polygonization conventions and cannot be directly XORed into a
   clean local change ground truth.
2. **No independent Shanghai benchmark.** No exhaustive, independently verified
   Shanghai change benchmark is currently available.
3. **Zero-shot ≠ local accuracy.** Shanghai results are zero-shot transfer
   observations, not local accuracy estimates.
4. **Seasonal / radiometric confounders.** Seasonal, radiometric, vegetation,
   and land-cover differences can produce high change probabilities.
5. **Small-change recall.** Small true building changes may still be missed.
6. **Residual registration error.** Residual registration errors remain
   relevant at sub-meter / pixel scale and can dominate segmentation
   differencing outputs.
7. **Human screening, not full automation.** The candidate-ranking system is
   intended for human screening rather than fully automated decisions; the
   human-review workflow is a prototype, not an operationally validated
   pipeline.

Additional technical caveats from the analyses:

- OpenCD direct-CD outputs are binary change maps (no gain/loss direction);
  gain/loss splitting does not apply to them.
- FC-Siam-diff overpredicts change fraction (~20.2%) in the Shanghai zero-shot
  setting; its outputs must be recalibrated before any operational use.
- The legacy ONNX files correspond to the WHU-generalized iter-14000 weights,
  not the iter-6000 local model — historical comparisons must label which
  weights were used (see [`model_provenance.md`](model_provenance.md)).
- Review cards existed in an earlier broken (black-crop) version; the current
  card generator includes QA, but the pre-fix cards were never published as
  results.
