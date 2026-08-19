"""Phase J - WHU-trained OpenCD model -> Shanghai Jan-Apr zero-shot inference.

Usage:
  python scripts/opencd_zero_shot_shanghai.py --config <opencd_cfg> --checkpoint <pth>
      [--tag fc_siam_diff] [--device cuda] [--stride 128]

The April image is resampled onto the January grid (bilinear) before inference;
no Shanghai GT is used. Outputs land in outputs/shanghai_zero_shot/<tag>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "open-cd"))

JAN_IMG = r"<LOCAL_SHANGHAI_DATA>/2026-01.tif"
APR_IMG = r"<LOCAL_SHANGHAI_DATA>/2026-04.tif"
OUT_ROOT = PROJECT_ROOT / "outputs" / "shanghai_zero_shot"

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
MEAN = MEAN.reshape(1, 3, 1, 1)
STD = STD.reshape(1, 3, 1, 1)


def load_model(config_path: str, checkpoint_path: str, device: str):
    from mmengine.config import Config
    from mmseg.apis import init_model

    cfg = Config.fromfile(config_path)
    return init_model(cfg, checkpoint_path, device=device)


def hann2d(size: int) -> np.ndarray:
    w = np.hanning(size).astype(np.float32)
    w = np.maximum(w, 0.05)
    return np.outer(w, w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.config, args.checkpoint, args.device)
    model.eval()

    # January grid = reference grid
    with rasterio.open(JAN_IMG) as jan_src:
        jan_h, jan_w = jan_src.height, jan_src.width
        jan_tr, jan_crs = jan_src.transform, jan_src.crs

    # April -> January grid (bilinear), probability-free RGB
    apr_aligned_path = out_dir / "april_aligned_jan_grid.tif"
    with rasterio.open(JAN_IMG) as jan_src, rasterio.open(APR_IMG) as apr_src:
        prof = jan_src.profile.copy()
        prof.update(count=3, dtype="uint8", compress="deflate")
        with rasterio.open(apr_aligned_path, "w", **prof) as dst:
            reproject(
                source=rasterio.band(apr_src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=apr_src.transform,
                src_crs=apr_src.crs,
                dst_transform=jan_src.transform,
                dst_crs=jan_src.crs,
                resampling=Resampling.bilinear,
            )
            reproject(
                source=rasterio.band(apr_src, 2),
                destination=rasterio.band(dst, 2),
                src_transform=apr_src.transform,
                src_crs=apr_src.crs,
                dst_transform=jan_src.transform,
                dst_crs=jan_src.crs,
                resampling=Resampling.bilinear,
            )
            reproject(
                source=rasterio.band(apr_src, 3),
                destination=rasterio.band(dst, 3),
                src_transform=apr_src.transform,
                src_crs=apr_src.crs,
                dst_transform=jan_src.transform,
                dst_crs=jan_src.crs,
                resampling=Resampling.bilinear,
            )

    tile, stride = args.tile, args.stride
    y_starts = list(range(0, jan_h - tile + 1, stride))
    if y_starts[-1] != jan_h - tile:
        y_starts.append(jan_h - tile)
    x_starts = list(range(0, jan_w - tile + 1, stride))
    if x_starts[-1] != jan_w - tile:
        x_starts.append(jan_w - tile)
    total = len(y_starts) * len(x_starts)

    prob_acc = np.zeros((jan_h, jan_w), dtype=np.float32)
    weight_acc = np.zeros((jan_h, jan_w), dtype=np.float32)
    weights = hann2d(tile)

    with rasterio.open(JAN_IMG) as jan_src, rasterio.open(apr_aligned_path) as apr_src:
        done = 0
        batch_a, batch_b, batch_wins = [], [], []

        def flush():
            nonlocal done
            if not batch_a:
                return
            xa = np.stack(batch_a, 0).astype(np.float32)
            xb = np.stack(batch_b, 0).astype(np.float32)
            ta = torch.from_numpy((xa - MEAN) / STD).to(args.device)
            tb = torch.from_numpy((xb - MEAN) / STD).to(args.device)
            x = torch.cat([ta, tb], dim=1)  # OpenCD SiamEncoderDecoder expects concat along channel
            with torch.no_grad():
                logits = model(x, mode="tensor")
                # ChangeFormer (SegformerHead) returns 1/4-resolution logits in tensor mode;
                # upsample to tile resolution so edge crops align with the accumulation grid.
                logits = torch.nn.functional.interpolate(
                    logits, size=(tile, tile), mode="bilinear", align_corners=False
                )
                prob = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            for p, win in zip(prob, batch_wins):
                hh = min(tile, jan_h - win.row_off)
                ww = min(tile, jan_w - win.col_off)
                crop = p[:hh, :ww]
                wc = weights[:hh, :ww]
                prob_acc[win.row_off : win.row_off + hh, win.col_off : win.col_off + ww] += crop * wc
                weight_acc[win.row_off : win.row_off + hh, win.col_off : win.col_off + ww] += wc
                done += 1
            batch_a.clear()
            batch_b.clear()
            batch_wins.clear()

        for y in y_starts:
            for x in x_starts:
                hh = min(tile, jan_h - y)
                ww = min(tile, jan_w - x)
                win = Window(x, y, ww, hh)
                a = jan_src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0)
                b = apr_src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0)
                pa = np.zeros((3, tile, tile), dtype=np.float32)
                pb = np.zeros((3, tile, tile), dtype=np.float32)
                pa[:, :hh, :ww] = a
                pb[:, :hh, :ww] = b
                batch_a.append(pa)
                batch_b.append(pb)
                batch_wins.append(win)
                if len(batch_a) >= args.batch:
                    flush()
                    print(f"  {done}/{total} tiles", end="\r")
        flush()

    prob_map = prob_acc / np.maximum(weight_acc, 1e-6)
    prob_map = np.asarray(prob_map).reshape(jan_h, jan_w).astype(np.float32)
    valid = weight_acc > 0
    prob_map[~valid] = 0
    mask = np.where(valid, (prob_map >= args.threshold).astype(np.uint8), 255).reshape(jan_h, jan_w).astype(np.uint8)

    prob_path = out_dir / "change_probability.tif"
    mask_path = out_dir / "change_mask.tif"
    with rasterio.open(JAN_IMG) as src:
        prof = src.profile.copy()
    prof.update(dtype="float32", count=1, compress="deflate", nodata=0.0, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(prob_path, "w", **prof) as dst:
        dst.write(prob_map, 1)
    prof.update(dtype="uint8", nodata=255)
    with rasterio.open(mask_path, "w", **prof) as dst:
        dst.write(mask, 1)

    stats = {
        "tag": args.tag,
        "tiles": total,
        "threshold": args.threshold,
        "change_fraction": round(float((mask == 1).sum() / max(int(valid.sum()), 1)), 6),
        "change_pixels": int((mask == 1).sum()),
        "valid_pixels": int(valid.sum()),
        "probability_path": str(prob_path),
        "mask_path": str(mask_path),
        "april_aligned": str(apr_aligned_path),
        "note": "Zero-shot geographic transfer; NO Shanghai metrics computed (GT Gate).",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
