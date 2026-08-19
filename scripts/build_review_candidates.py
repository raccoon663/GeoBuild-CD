"""Phases Q/R/S/T - candidate change objects from ChangeFormer + other models.

Extracts candidate objects from:
  - ChangeFormer binary change map (primary candidates, all_candidates.gpkg)
  - Original App (iter-14000) C0.7 change NOT covered by CF/FC-Siam (oa_only)
  - FC-Siam change NOT covered by CF/iter-14000 (fc_only)
  - negative-control random points (no-change areas)

Computes per-object size / geometry / probability / overlaps / boundary
distance / building context / spectral diff / compactness.

Outputs: outputs/review_candidates/{all_candidates,oa_only,fc_only,negative_controls}.gpkg|.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from shapely.geometry import Point, shape
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

OUT = PROJECT_ROOT / "outputs" / "review_candidates"
CFG = yaml.safe_load((PROJECT_ROOT / "configs" / "candidate_ranking.yaml").read_text(encoding="utf-8"))

CF_PROB = PROJECT_ROOT / "outputs" / "shanghai_zero_shot" / "changeformer" / "change_probability.tif"
CF_MASK = PROJECT_ROOT / "outputs" / "shanghai_zero_shot" / "changeformer" / "change_mask.tif"
JAN_IMG = r"<LOCAL_SHANGHAI_DATA>/2026-01.tif"
APR_RGB = PROJECT_ROOT / "outputs" / "shanghai_zero_shot" / "changeformer" / "april_aligned_jan_grid.tif"
JAN_PROB = PROJECT_ROOT / "outputs" / "segmentation" / "2026-01" / "building_probability.tif"
APR_PROB_ALIGNED = PROJECT_ROOT / "outputs" / "old_change_baseline" / "april_probability_aligned_jan_grid.tif"
LEGACY_6000 = PROJECT_ROOT / "outputs" / "old_change_baseline" / "C0.7_grid_change.tif"
LEGACY_14000 = PROJECT_ROOT / "outputs" / "original_app_baseline" / "C0.7_grid_change.tif"
FC_SIAM_MASK = PROJECT_ROOT / "outputs" / "shanghai_zero_shot" / "fc_siam_diff" / "change_mask.tif"

PIX_AREA_M2 = 0.25


def read_band(path, band=1):
    with rasterio.open(path) as src:
        return src.read(band)


def main() -> None:
    import geopandas as gpd
    from rasterio import features

    OUT.mkdir(parents=True, exist_ok=True)
    prob = read_band(CF_PROB).astype(np.float32)
    cf = read_band(CF_MASK) == 1
    with rasterio.open(CF_MASK) as src:
        transform = src.transform
        crs = src.crs

    jan_valid = read_band(JAN_PROB) > 0
    apr_rgb_full = rasterio.open(APR_RGB).read([1, 2, 3])
    valid = jan_valid & (apr_rgb_full.max(axis=0) > 5)
    del jan_valid

    jan_img = rasterio.open(JAN_IMG).read([1, 2, 3]).astype(np.float32)
    apr_img = apr_rgb_full.astype(np.float32)
    del apr_rgb_full
    spec = np.empty((jan_img.shape[1], jan_img.shape[2]), dtype=np.float32)
    for r0 in range(0, jan_img.shape[1], 2048):
        r1 = min(r0 + 2048, jan_img.shape[1])
        spec[r0:r1] = np.abs(apr_img[:, r0:r1] - jan_img[:, r0:r1]).mean(axis=0)
    del jan_img, apr_img

    jan_bp = read_band(JAN_PROB).astype(np.float32)
    apr_bp = read_band(APR_PROB_ALIGNED).astype(np.float32)
    build_mask = (jan_bp >= 0.5) | (apr_bp >= 0.5)

    import cv2

    # distance to nearest building (cv2, memory-light); 0 inside buildings
    dist_build = cv2.distanceTransform(
        (build_mask).astype(np.uint8), cv2.DIST_L2, 5
    ).astype(np.float32)
    dist_build[build_mask] = 0.0
    building_edge = (dist_build > 0) & (dist_build <= 4)
    dist_bnd = cv2.distanceTransform(
        (valid).astype(np.uint8), cv2.DIST_L2, 5
    ).astype(np.float32)

    leg6 = read_band(LEGACY_6000) > 0
    leg14 = read_band(LEGACY_14000) > 0
    fc = read_band(FC_SIAM_MASK) == 1

    size_groups = CFG["size_groups_m2"]

    def objects_from_label(lbl, n, prefix, use_prob):
        rows = []
        size_counts = {g[2]: 0 for g in size_groups}
        slices = ndimage.find_objects(lbl)
        for i in range(1, n + 1):
            sl = slices[i - 1]
            if sl is None:
                continue
            crop = lbl[sl]
            yy, xx = np.nonzero(crop == i)
            yy = yy + sl[0].start
            xx = xx + sl[1].start
            area_px = yy.size
            area_m2 = area_px * PIX_AREA_M2
            size_group = None
            for g in size_groups:
                lo, hi, name = g
                if area_m2 >= lo and (hi is None or area_m2 < hi):
                    size_counts[name] += 1
                    size_group = name
                    break
            pv = use_prob[yy, xx]
            r0, r1 = int(yy.min()), int(yy.max())
            c0, c1 = int(xx.min()), int(xx.max())
            m = (lbl[r0 : r1 + 1, c0 : c1 + 1] == i).astype(np.uint8)
            crop_tr = rasterio.windows.transform(
                rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1), transform
            )
            # features.shapes(..., transform=...) already returns coordinates in
            # world space (EPSG:4326). Applying a second affine_transform here
            # (previous bug) collapsed every polygon into a ~1e-6 deg cluster
            # near the raster origin (invalid canvas edge). Do NOT re-transform.
            # Second bug: ndimage.label uses 8-connectivity while features.shapes
            # defaults to 4-connectivity, so diagonal-connected components were
            # split into several polygons and next() kept only a 1-px fragment.
            # Use connectivity=8 and union all returned shapes.
            poly = unary_union(
                [
                    shape(g)
                    for g, _ in features.shapes(
                        m, mask=m, transform=crop_tr, connectivity=8
                    )
                ]
            )
            del crop, m
            rows.append(
                {
                    "candidate_id": f"{prefix}_{i:06d}",
                    "area_m2": round(area_m2, 2),
                    "size_group": size_group,
                    "x": round(float(poly.centroid.x), 7),
                    "y": round(float(poly.centroid.y), 7),
                    "changeformer_mean_prob": round(float(pv.mean()), 4),
                    "changeformer_max_prob": round(float(pv.max()), 4),
                    "legacy_iter6000_overlap": round(float((leg6[yy, xx]).sum()) / area_px, 4),
                    "legacy_iter14000_overlap": round(float((leg14[yy, xx]).sum()) / area_px, 4),
                    "fc_siam_overlap": round(float((fc[yy, xx]).sum()) / area_px, 4),
                    "boundary_distance_px": round(float(dist_bnd[yy, xx].min()), 2),
                    "distance_to_building_m": round(float(dist_build[yy, xx].min()) * 0.5, 2),
                    "jan_building_overlap": round(float((jan_bp[yy, xx] >= 0.5).mean()), 4),
                    "apr_building_overlap": round(float((apr_bp[yy, xx] >= 0.5).mean()), 4),
                    "jan_building_mean_prob": round(float(jan_bp[yy, xx].mean()), 4),
                    "apr_building_mean_prob": round(float(apr_bp[yy, xx].mean()), 4),
                    "mean_spectral_diff": round(float(spec[yy, xx].mean()), 2),
                    "building_edge_fraction": round(float(building_edge[yy, xx].mean()), 4),
                    "compactness": round(
                        float(4 * np.pi * poly.area / max(poly.length**2, 1e-12)), 4
                    ),
                    "geometry": poly,
                }
            )
        return rows, size_counts

    def save(gdf_rows, name, stats):
        gdf = gpd.GeoDataFrame(gdf_rows, crs=crs).set_geometry("geometry")
        gdf.to_file(OUT / f"{name}.gpkg", driver="GPKG", encoding="utf-8")
        cols = [c for c in gdf.columns if c != "geometry"]
        gdf[cols].to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
        (OUT / f"{name}_size_groups.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return gdf

    lbl_cf, n_cf = ndimage.label((cf & valid).astype(np.uint8), structure=np.ones((3, 3)))
    rows_cf, size_cf = objects_from_label(lbl_cf, n_cf, "CF", prob)
    del lbl_cf
    save(rows_cf, "all_candidates", {"n": n_cf, "size_groups": size_cf})
    print("CF candidates:", n_cf, size_cf)

    oa_only = (leg14 & valid) & ~cf & ~fc
    lbl_oa, n_oa = ndimage.label(oa_only.astype(np.uint8), structure=np.ones((3, 3)))
    rows_oa, size_oa = objects_from_label(lbl_oa, n_oa, "OA", prob)
    del lbl_oa, oa_only
    save(rows_oa, "oa_only", {"n": n_oa, "size_groups": size_oa})
    print("OA-only objects:", n_oa, size_oa)

    fc_only = (fc & valid) & ~cf & ~leg14
    lbl_fc, n_fc = ndimage.label(fc_only.astype(np.uint8), structure=np.ones((3, 3)))
    rows_fc, size_fc = objects_from_label(lbl_fc, n_fc, "FS", prob)
    del lbl_fc, fc_only
    save(rows_fc, "fc_only", {"n": n_fc, "size_groups": size_fc})
    print("FC-only objects:", n_fc, size_fc)

    rng = np.random.default_rng(CFG["stratified_review"]["seed"])
    no_change = valid & ~cf & ~leg14 & ~fc & (dist_bnd >= 64)
    flat = no_change.ravel()
    pick = []
    tries = 0
    while len(pick) < 40 and tries < 2_000_000:
        f = int(rng.integers(0, flat.size))
        if flat[f]:
            pick.append(f)
        tries += 1
    print("negative controls sampled:", len(pick))
    pts = []
    for k, flat in enumerate(pick):
        row_, col_ = divmod(int(flat), valid.shape[1])
        lon, lat = rasterio.transform.xy(transform, row_, col_)
        pts.append(
            {
                "candidate_id": f"NEG_{k + 1:03d}",
                "area_m2": 0.0,
                "size_group": "negative_control",
                "x": round(float(lon), 7),
                "y": round(float(lat), 7),
                "changeformer_mean_prob": round(float(prob[row_, col_]), 4),
                "changeformer_max_prob": round(float(prob[row_, col_]), 4),
                "legacy_iter6000_overlap": 0.0,
                "legacy_iter14000_overlap": 0.0,
                "fc_siam_overlap": 0.0,
                "boundary_distance_px": round(float(dist_bnd[row_, col_]), 2),
                "distance_to_building_m": round(float(dist_build[row_, col_]) * 0.5, 2),
                "jan_building_overlap": round(float(jan_bp[row_, col_] >= 0.5), 4),
                "apr_building_overlap": round(float(apr_bp[row_, col_] >= 0.5), 4),
                "jan_building_mean_prob": round(float(jan_bp[row_, col_]), 4),
                "apr_building_mean_prob": round(float(apr_bp[row_, col_]), 4),
                "mean_spectral_diff": round(float(spec[row_, col_]), 2),
                "building_edge_fraction": round(float(building_edge[row_, col_]), 4),
                "compactness": None,
                "geometry": Point(lon, lat),
            }
        )
    save(pts, "negative_controls", {"n": len(pts)})
    print("negative controls:", len(pts))


if __name__ == "__main__":
    main()
