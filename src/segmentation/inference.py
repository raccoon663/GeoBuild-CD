"""Tiled GPU inference engine for large GeoTIFFs.

Streams 512x512 patches (with optional overlap) through the pure-PyTorch
SegFormer-B5 model, writes georeferenced float32 probability GeoTIFFs and
uint8 building masks, and optionally vectorizes the mask.

Channel convention: RGB (bands 1-3), ImageNet normalization embedded in
`SegFormerB5.normalize`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio import features
from rasterio.windows import Window
from tqdm import tqdm


MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)


def _window_starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _hann2d(size: int) -> np.ndarray:
    w = np.hanning(size).astype(np.float32)
    w = np.maximum(w, 0.05)
    return np.outer(w, w)


def predict_raster(
    image_path,
    model: torch.nn.Module,
    prob_path=None,
    mask_path=None,
    tile_size: int = 512,
    stride: int | None = None,
    threshold: float = 0.5,
    batch_size: int = 4,
    device: str = "cuda",
    progress: bool = True,
) -> dict:
    """Run tiled inference and write probability + mask GeoTIFFs.

    Returns stats dict.  At least one of prob_path / mask_path must be set.
    """
    stride = stride or tile_size
    assert 0 < stride <= tile_size
    overlap = tile_size - stride
    model = model.to(device).eval()
    with torch.no_grad():
        torch.set_grad_enabled(False)

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        y_starts = _window_starts(height, tile_size, stride)
        x_starts = _window_starts(width, tile_size, stride)
        total = len(y_starts) * len(x_starts)

        prob_profile = None
        mask_profile = None
        dst_prob = None
        dst_mask = None
        if prob_path:
            prob_profile = src.profile.copy()
            prob_profile.update(
                driver="GTiff", dtype="float32", count=1, compress="deflate",
                nodata=None, tiled=True, blockxsize=256, blockysize=256,
            )
            Path(prob_path).parent.mkdir(parents=True, exist_ok=True)
            dst_prob = rasterio.open(prob_path, "w", **prob_profile)
        if mask_path:
            mask_profile = src.profile.copy()
            mask_profile.update(
                driver="GTiff", dtype="uint8", count=1, compress="deflate",
                nodata=255, tiled=True, blockxsize=256, blockysize=256,
            )
            Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
            dst_mask = rasterio.open(mask_path, "w", **mask_profile)

        # accumulation buffers only when overlapping
        prob_acc = np.zeros((height, width), dtype=np.float32) if overlap else None
        weight_acc = np.zeros((height, width), dtype=np.float32) if overlap else None
        valid_acc = np.zeros((height, width), dtype=np.bool_) if overlap else None
        weights = _hann2d(tile_size)

        batch_arrays = []
        batch_valids = []
        batch_windows = []
        done = 0
        pbar = tqdm(total=total, disable=not progress, desc="inference")

        def flush():
            nonlocal done
            if not batch_arrays:
                return
            xb = np.stack(batch_arrays, axis=0).astype(np.float32)
            xt = torch.from_numpy((xb - MEAN) / STD).to(device)
            with torch.no_grad():
                logits = model(xt)
                prob = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            for prob_patch, valid_crop, win in zip(prob, batch_valids, batch_windows):
                h = min(tile_size, height - win.row_off)
                w = min(tile_size, width - win.col_off)
                crop = prob_patch[:h, :w]
                valid_crop = valid_crop[:h, :w]
                if overlap:
                    wcrop = weights[:h, :w]
                    prob_acc[win.row_off : win.row_off + h, win.col_off : win.col_off + w] += crop * wcrop * valid_crop
                    weight_acc[win.row_off : win.row_off + h, win.col_off : win.col_off + w] += wcrop * valid_crop
                else:
                    crop = crop * valid_crop
                    if dst_prob is not None:
                        dst_prob.write(crop, 1, window=win)
                    if dst_mask is not None:
                        mask = np.where(valid_crop > 0, ((crop >= threshold) & (crop > 0)).astype(np.uint8), 255)
                        dst_mask.write(mask, 1, window=win)
                done += 1
                pbar.update(1)
            batch_arrays.clear()
            batch_valids.clear()
            batch_windows.clear()

        for y in y_starts:
            for x in x_starts:
                win_h = min(tile_size, height - y)
                win_w = min(tile_size, width - x)
                win = Window(x, y, win_w, win_h)
                data = src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0)
                patch = np.zeros((3, tile_size, tile_size), dtype=np.float32)
                patch[:, :win_h, :win_w] = data
                valid_crop = np.zeros((tile_size, tile_size), dtype=np.float32)
                valid_crop[:win_h, :win_w] = (data > 5).any(axis=0)
                batch_arrays.append(patch)
                batch_valids.append(valid_crop)
                batch_windows.append(win)
                if len(batch_arrays) >= batch_size:
                    flush()
        flush()
        pbar.close()

        if overlap and prob_acc is not None:
            prob_map = prob_acc / np.maximum(weight_acc, 1e-6)
            prob_map[weight_acc <= 0] = 0
            if dst_prob is not None:
                dst_prob.write(prob_map[None], 1)
            if dst_mask is not None:
                mask = np.where(weight_acc > 0, ((prob_map >= threshold) & (prob_map > 0)).astype(np.uint8), 255)
                dst_mask.write(mask[None], 1)

        if dst_prob is not None:
            dst_prob.close()
        if dst_mask is not None:
            dst_mask.close()

    return {
        "image": str(image_path),
        "height": height,
        "width": width,
        "tiles": total,
        "tile_size": tile_size,
        "stride": stride,
        "threshold": threshold,
        "probability_path": str(prob_path) if prob_path else None,
        "mask_path": str(mask_path) if mask_path else None,
    }


def mask_to_polygons(
    mask_path,
    out_gpkg: str | None = None,
    min_area_m2: float | None = None,
    simplify_tolerance_m: float | None = None,
    foreground_values=(1, 255),
) -> dict:
    """Vectorize a building mask GeoTIFF into polygons (georeferenced)."""
    import geopandas as gpd
    from shapely.geometry import shape

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        pixel_area = abs(transform.a * transform.e)
        fg = np.isin(mask, foreground_values)
        results = features.shapes(mask.astype(np.uint8), mask=fg, transform=transform)
        polys = []
        for geom, value in results:
            if value == 0:
                continue
            g = shape(geom)
            polys.append(g)
        gdf = gpd.GeoDataFrame({"geometry": polys}, crs=crs)
        if len(gdf) == 0:
            gdf = gpd.GeoDataFrame({"geometry": []}, crs=crs)
        if simplify_tolerance_m:
            # simplify in projected meters if needed
            pass
        if min_area_m2 and len(gdf):
            if str(crs).upper() == "EPSG:4326":
                # approximate: degrees -> meters via mean latitude
                lat = float(gdf.geometry.centroid.y.mean())
                scale = 111_320.0 * abs(np.cos(np.deg2rad(lat)))
                gdf = gdf[gdf.geometry.area * scale * 111_320.0 >= min_area_m2]
            else:
                gdf = gdf[gdf.geometry.area >= min_area_m2]
        if out_gpkg:
            Path(out_gpkg).parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(out_gpkg, driver="GPKG", encoding="utf-8")
        return {
            "polygon_count": int(len(gdf)),
            "output": str(out_gpkg) if out_gpkg else None,
            "pixel_area_m2": float(pixel_area),
        }
