"""Run SegFormer-B5 building inference on a monthly image.

Usage:
  python scripts/predict_month.py --month 2026-01 --out-dir outputs/segmentation/2026-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.inference import predict_raster, mask_to_polygons
from src.segmentation.segformer_mit import build_segformer_b5_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--vectorize", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    month_cfg = cfg["data"]["months"][args.month]
    image_path = month_cfg["image"]
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "outputs" / "segmentation" / args.month
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_segformer_b5_from_checkpoint(str(PROJECT_ROOT / "checkpoints" / "segformer_b5_zhelin_local_best_20260624.pth"))
    stats = predict_raster(
        image_path,
        model,
        prob_path=out_dir / "building_probability.tif",
        mask_path=out_dir / "building_mask.tif",
        stride=args.stride or cfg["experiment"]["patch_size"],
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )
    if args.vectorize:
        v = mask_to_polygons(
            out_dir / "building_mask.tif",
            out_gpkg=out_dir / "building_footprints.gpkg",
            min_area_m2=cfg["experiment"]["min_building_area_m2"],
        )
        stats.update(v)
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
