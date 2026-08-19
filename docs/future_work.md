# Future Work

The project is paused at the HUMAN REVIEW GATE (see
[`reports/HUMAN_REVIEW_GATE.md`](../reports/HUMAN_REVIEW_GATE.md)): research
prototype complete, before local supervised adaptation and exhaustive human
validation.

Planned next steps (in priority order):

1. Build a small, exhaustively reviewed Shanghai benchmark.
2. Collect hard negatives from seasonal false positives.
3. Fine-tune ChangeFormer with limited local supervision.
4. Evaluate label efficiency under 50 / 100 / 200 / 500 local patches.
5. Measure small-change recall on an independent held-out AOI.
6. Validate Precision@K and candidate-ranking efficiency.
7. Improve the human-review interface.

> Local adaptation is intentionally left as a future phase rather than being
> performed on the current review candidates — the review pool must stay clean
> so Precision@K after human review remains a valid estimate of the unadapted
> screening pipeline.
