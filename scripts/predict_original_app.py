"""Phase N - reproduce the Original Building Change Detector App segmentation.

Runs the WHU-generalized SegFormer-B5 (iter-14000) on Shanghai January and
April, writing probability + mask GeoTIFFs into outputs/original_app_baseline/.
These feed the Legacy-B differencing pipeline (same as Phase D, but with the
weights the packaged app actually shipped).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.inference import predict_raster
from src.segmentation.segformer_mit import build_segformer_b5_from_checkpoint

CKPT = r"<MODEL_ROOT>/segformer_b5_whu_generalized_best_mIoU_iter_14000.pth"
JAN_IMG = r"<LOCAL_SHANGHAI_DATA>/2026-01.tif"
APR_IMG = r"<LOCAL_SHANGHAI_DATA>/2026-04.tif"
OUT = PROJECT_ROOT / "outputs" / "original_app_baseline"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_segformer_b5_from_checkpoint(CKPT)
    summary = {"checkpoint": CKPT, "months": {}}
    for name, img in [("january", JAN_IMG), ("april", APR_IMG)]:
        stats = predict_raster(
            img,
            model,
            prob_path=OUT / f"{name}_building_probability.tif",
            mask_path=OUT / f"{name}_building_mask.tif",
            stride=512,
            batch_size=4,
            device="cuda",
        )
        summary["months"][name] = stats
        print(name, stats)
    (OUT / "predict_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
