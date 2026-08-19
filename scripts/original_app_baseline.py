"""Phase N - Legacy-B: Original Building Change Detector App baseline.

Uses the WHU-generalized SegFormer-B5 (iter-14000) probability maps produced by
`predict_original_app.py` and runs the SAME differencing pipeline as
`old_change_baseline.py` (Legacy-A, iter-6000).

Outputs: outputs/original_app_baseline/ + reports/original_app_baseline.md/.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from old_change_baseline import (  # noqa: E402
    build_masks,
    clean_component,
    load_prob,
    shift_prob,
    summarize,
    warp_prob_to,
)

JAN_PROB = PROJECT_ROOT / "outputs" / "original_app_baseline" / "january_building_probability.tif"
APR_PROB = PROJECT_ROOT / "outputs" / "original_app_baseline" / "april_building_probability.tif"
OUT = PROJECT_ROOT / "outputs" / "original_app_baseline"
REPORTS = PROJECT_ROOT / "reports"

PC_DX = -1.61
PC_DY = 0.17
PIXEL_AREA_M2 = 0.25
MIN_OBJECT_PIXELS = int(30.0 / PIXEL_AREA_M2)
HOLE_FILL_PIXELS = int(20.0 / PIXEL_AREA_M2)


def write_mask(mask: np.ndarray, tif: Path, ref: Path) -> None:
    with rasterio.open(ref) as r:
        profile = r.profile.copy()
    profile.update(dtype="uint8", count=1, compress="deflate", nodata=255)
    arr = np.full(mask.shape, 255, dtype=np.uint8)
    arr[mask] = 1
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(arr, 1)


def write_change(g: np.ndarray, l: np.ndarray, tif: Path, ref: Path) -> None:
    with rasterio.open(ref) as r:
        profile = r.profile.copy()
    profile.update(dtype="uint8", count=1, compress="deflate", nodata=255)
    arr = np.zeros(g.shape, dtype=np.uint8)
    arr[g] = 1
    arr[l] = 2
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(arr, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jan_p, jan_tr, jan_crs = load_prob(JAN_PROB)
    apr_p, apr_tr, apr_crs = load_prob(APR_PROB)
    aligned_path = OUT / "april_probability_aligned_jan_grid.tif"
    apr_aligned = warp_prob_to(apr_p, apr_tr, apr_crs, jan_tr, jan_crs, jan_p.shape, aligned_path)
    apr_aligned = np.clip(apr_aligned, 0.0, 1.0)
    del apr_p

    summary = {}
    for label, apr_use in (("grid", apr_aligned), ("shiftcorr", None)):
        if apr_use is None:
            apr_use = shift_prob(apr_aligned, PC_DX, PC_DY)
        valid = (jan_p > 0.0) & (apr_use > 0.0)
        for key, masks in build_masks(jan_p, apr_use, valid).items():
            name = f"{key}_{label}"
            write_mask(masks["gain"], OUT / f"{name}_gain.tif", aligned_path)
            write_mask(masks["loss"], OUT / f"{name}_loss.tif", aligned_path)
            summary[name] = summarize(masks, name)
        if label == "shiftcorr":
            del apr_use

    for t in (0.5, 0.7):
        for label in ("grid", "shiftcorr"):
            src_key = f"B{t:g}_{label}"
            if src_key not in summary:
                continue
            with rasterio.open(OUT / f"{src_key}_gain.tif") as src:
                g0 = src.read(1) == 1
            with rasterio.open(OUT / f"{src_key}_loss.tif") as src:
                l0 = src.read(1) == 1
            cg = clean_component(g0, MIN_OBJECT_PIXELS, HOLE_FILL_PIXELS)
            cl = clean_component(l0, MIN_OBJECT_PIXELS, HOLE_FILL_PIXELS)
            name = f"C{t:g}_{label}"
            write_mask(cg, OUT / f"{name}_gain.tif", aligned_path)
            write_mask(cl, OUT / f"{name}_loss.tif", aligned_path)
            summary[name] = summarize({"gain": cg, "loss": cl}, name)
            del g0, l0, cg, cl

    for key in ("A_grid", "B0.5_grid", "B0.7_grid", "B0.8_grid", "C0.5_grid", "C0.7_grid", "C0.7_shiftcorr"):
        if key not in summary:
            continue
        with rasterio.open(OUT / f"{key}_gain.tif") as src:
            g = src.read(1) == 1
        with rasterio.open(OUT / f"{key}_loss.tif") as src:
            l = src.read(1) == 1
        write_change(g, l, OUT / f"{key}_change.tif", aligned_path)
        del g, l

    (REPORTS / "original_app_baseline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Original App Baseline（Legacy-B, SegFormer iter-14000 differencing）",
        "",
        "- 模型：WHU-generalized SegFormer-B5（`best_mIoU_iter_14000.pth`）——原 Building Change Detector App 打包 ONNX 的实际权重（`model_provenance.md` §5）。",
        "- January/April probability 由 `scripts/predict_original_app.py` 生成；April 重采样到 Jan 网格（derivative），与 Legacy-A 相同 pipeline（A raw / B0.5/0.7/0.8 / C0.5/0.7 / shiftcorr）。",
        "- **不计算 Shanghai 指标（GT Gate 前禁止）**。",
        "",
        "## 汇总统计",
        "",
        "| Variant | Changed px | Changed area (ha) | Gain objs | Loss objs | Gain area (ha) | Loss area (ha) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, s in summary.items():
        md.append(
            f"| {key} | {s['changed_pixels']:,} | {s['changed_area_m2'] / 1e4:.2f} | "
            f"{s['gain']['count']:,} | {s['loss']['count']:,} | {s['gain']['area_m2'] / 1e4:.2f} | {s['loss']['area_m2'] / 1e4:.2f} |"
        )
    md += [
        "",
        "## 与 Legacy-A（iter-6000）对比",
        "",
        "- 见 `reports/old_change_baseline.md`；两者使用同一 pipeline 与同一 Jan/Apr 影像，仅模型权重不同（iter-6000 本地 vs iter-14000 WHU-generalized）。",
        "- 最终 legacy 对比（Phase P）：Legacy local（iter-6000）vs Original App（iter-14000）vs Direct CD（ChangeFormer / FC-Siam）。",
        "",
        "输出：`outputs/original_app_baseline/*`（prob/mask/aligned/change tifs + stats.json）。",
        "",
    ]
    (REPORTS / "original_app_baseline.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/original_app_baseline.md")
    for key in ("A_grid", "B0.7_grid", "C0.7_grid", "C0.7_shiftcorr"):
        s = summary[key]
        print(
            f"{key}: changed={s['changed_area_m2']/1e4:.2f} ha | gain objs={s['gain']['count']} | "
            f"loss objs={s['loss']['count']}"
        )


if __name__ == "__main__":
    main()
