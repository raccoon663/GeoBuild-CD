# Third-Party Notices

This repository contains original research code and small derived
configurations. The components below are third-party and are attributed here
with their upstream licenses. **No third-party source tree is vendored** in
this repository; upstream projects are referenced and installed separately.

## Upstream projects (referenced, not vendored)

| Component | Upstream | License | Used for |
| --------- | -------- | ------- | -------- |
| OpenCD | https://github.com/likyoo/open-cd | Apache-2.0 | Direct change detection models (FC-Siam-diff, ChangeFormer), training/validation tooling |
| MMEngine | https://github.com/open-mmlab/mmengine | Apache-2.0 | OpenCD runtime |
| MMCV | https://github.com/open-mmlab/mmcv | Apache-2.0 | OpenCD runtime |
| MMSegmentation | https://github.com/open-mmlab/mmsegmentation | Apache-2.0 | SegFormer head / dataset tooling used by OpenCD |
| mmdet / mmpretrain | https://github.com/open-mmlab | Apache-2.0 | OpenCD dependency stack |
| TorchGeo | https://github.com/microsoft/torchgeo | MIT | Geospatial dataset management |
| Segment-Geospatial (SamGeo) | https://github.com/opengeos/segment-geospatial | MIT | Annotation aid (exploratory only) |

## Derived files in this repository

The OpenCD run configs under `configs/changeformer/` and `configs/fcsn/` are
derived from OpenCD configuration conventions (Apache-2.0); they are kept as
small self-contained files with `data_root` pointing at the `<WHU_CD_ROOT>`
placeholder (see `docs/data_and_splits.md`).

## Pretrained weights

- `mit_b0` initial weights: OpenMMLab MMSegmentation pretrained checkpoints
  (downloaded from the official `download.openmmlab.com` URL referenced in
  `configs/changeformer/`).
- No trained checkpoints are redistributed in this repository.

## Datasets (not redistributed)

| Dataset | Source | Note |
| ------- | ------ | ---- |
| WHU-CD | Official WHU building change detection dataset | Download from the official source; only preparation scripts and split configs are included here |
| Shanghai local imagery / labels | Restricted local data | Not redistributed due to data-access restrictions (see `docs/data_and_splits.md`) |

## License of this repository

The original code in this repository is made available under the
[Apache License 2.0](LICENSE), consistent with the Apache-2.0 dependencies it
builds on. This is a maintainer decision; if you plan to redistribute, review
`LICENSE` and the upstream licenses above.
