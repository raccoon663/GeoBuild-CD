"""Phase D - Old SegFormer difference baseline (segmentation-then-difference).

Builds change-candidate maps from the existing iter-6000 SegFormer-B5
probability maps (January & April) with several post-processing variants:

  A         raw mask XOR (p>=0.5)
  B{0.5,0.7,0.8}  confidence-filtered differencing
            gain = Apr>=t & Jan<=1-t ; loss = Jan>=t & Apr<=1-t
  C{t}      object-filtered differencing (remove tiny components, holes, opening)

Two alignment variants are produced:
  grid      April resampled onto the January grid (required baseline)
  shiftcorr additionally applies the registration-audit PC correction
            (shift April by dx=-1.61, dy=+0.17 Jan px) before differencing.

No Shanghai change metrics are computed (GT Gate).
Outputs: outputs/old_change_baseline/ + reports/old_change_baseline.md/.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

JAN_PROB = PROJECT_ROOT / "outputs" / "segmentation" / "2026-01" / "building_probability.tif"
APR_PROB = PROJECT_ROOT / "outputs" / "segmentation" / "2026-04" / "building_probability.tif"
OUT = PROJECT_ROOT / "outputs" / "old_change_baseline"
REPORTS = PROJECT_ROOT / "reports"

PC_DX = -1.61  # correction applied to April (Jan px), from jan_apr_registration.json
PC_DY = 0.17

PIXEL_AREA_M2 = 0.5 * 0.5  # ~0.5 m x 0.5 m at Jan GSD
MIN_OBJECT_PIXELS = int(30.0 / PIXEL_AREA_M2)  # 30 m2
HOLE_FILL_PIXELS = int(20.0 / PIXEL_AREA_M2)  # 20 m2


def load_prob(path: Path) -> tuple[np.ndarray, rasterio.Affine, object]:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        return data, src.transform, src.crs


def warp_prob_to(src, src_transform, src_crs, dst_transform, dst_crs, shape, out_path: Path) -> np.ndarray:
    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "deflate",
        "nodata": 0.0,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        reproject(
            source=src,
            destination=rasterio.band(dst, 1),
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=0.0,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=0.0,
            resampling=Resampling.bilinear,
        )
    with rasterio.open(out_path) as src:
        return src.read(1).astype(np.float32)


def shift_prob(data: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Apply a residual shift to align April onto January (moves content by -s)."""
    return ndimage.shift(data, shift=(-dy, -dx), order=1, mode="constant", cval=0.0)


def build_masks(jan_p: np.ndarray, apr_p: np.ndarray, valid: np.ndarray) -> dict:
    out = {}
    # A: raw XOR at 0.5
    gain = (apr_p >= 0.5) & (jan_p < 0.5) & valid
    loss = (jan_p >= 0.5) & (apr_p < 0.5) & valid
    out["A"] = {"gain": gain, "loss": loss}
    for t in (0.5, 0.7, 0.8):
        gain = (apr_p >= t) & (jan_p <= 1.0 - t) & valid
        loss = (jan_p >= t) & (apr_p <= 1.0 - t) & valid
        out[f"B{t:g}"] = {"gain": gain, "loss": loss}
    return out


def clean_component(mask: np.ndarray, min_pixels: int, hole_fill: int) -> np.ndarray:
    """Object filtering: fill small holes, binary opening, remove tiny objects."""
    m = mask.astype(np.uint8)
    if hole_fill > 0:
        filled = ndimage.binary_fill_holes(m)
        lbl_f, nf = ndimage.label(filled)
        if nf:
            ids, counts = np.unique(lbl_f[filled], return_counts=True)
            big = ids[counts > hole_fill]
            if len(big):
                filled[np.isin(lbl_f, big)] = False
        del lbl_f
        m = (filled | mask.astype(bool)).astype(np.uint8)
        del filled
    m = ndimage.binary_opening(m, structure=np.ones((3, 3))).astype(np.uint8)
    lbl, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(m, dtype=bool)
    ids, counts = np.unique(lbl[m.astype(bool)], return_counts=True)
    small = ids[counts < min_pixels]
    out = (m > 0)
    if len(small):
        out[np.isin(lbl, small)] = False
    del lbl, m
    return out


def component_stats(mask: np.ndarray) -> dict:
    lbl, n = ndimage.label(mask.astype(np.uint8))
    if n == 0:
        return {"count": 0, "pixels": int(mask.sum()), "area_m2": 0.0, "size_bins": {}}
    ids, counts = np.unique(lbl[mask], return_counts=True)
    sizes = counts.astype(np.int64)
    del lbl
    bins = {
        "tiny_<30m2": int((sizes < 120).sum()),
        "30-100m2": int(((sizes >= 120) & (sizes < 400)).sum()),
        "100-500m2": int(((sizes >= 400) & (sizes < 2000)).sum()),
        "500m2-0.5ha": int(((sizes >= 2000) & (sizes < 20000)).sum()),
        ">=0.5ha": int((sizes >= 20000).sum()),
    }
    return {
        "count": int(n),
        "pixels": int(mask.sum()),
        "area_m2": round(float(mask.sum() * PIXEL_AREA_M2), 1),
        "size_bins": bins,
        "median_px": int(np.median(sizes)) if n else 0,
        "max_px": int(sizes.max()) if n else 0,
    }


def summarize(masks: dict, name: str) -> dict:
    gain = masks["gain"]
    loss = masks["loss"]
    return {
        "variant": name,
        "gain": component_stats(gain),
        "loss": component_stats(loss),
        "changed_pixels": int((gain | loss).sum()),
        "changed_area_m2": round(float((gain | loss).sum() * PIXEL_AREA_M2), 1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jan_p, jan_tr, jan_crs = load_prob(JAN_PROB)
    apr_p, apr_tr, apr_crs = load_prob(APR_PROB)

    # derivative aligned data: April probability on January grid
    aligned_path = OUT / "april_probability_aligned_jan_grid.tif"
    apr_aligned = warp_prob_to(
        apr_p, apr_tr, apr_crs, jan_tr, jan_crs, jan_p.shape, aligned_path
    )
    apr_aligned = np.clip(apr_aligned, 0.0, 1.0)
    del apr_p

    def write_mask(mask: np.ndarray, tif: Path, aligned_path: Path) -> None:
        with rasterio.open(aligned_path) as ref:
            profile = ref.profile.copy()
        profile.update(dtype="uint8", count=1, compress="deflate", nodata=255)
        arr = np.full(mask.shape, 255, dtype=np.uint8)
        arr[mask] = 1
        with rasterio.open(tif, "w", **profile) as dst:
            dst.write(arr, 1)

    def write_change(g: np.ndarray, l: np.ndarray, tif: Path, aligned_path: Path) -> None:
        with rasterio.open(aligned_path) as ref:
            profile = ref.profile.copy()
        profile.update(dtype="uint8", count=1, compress="deflate", nodata=255)
        arr = np.zeros(g.shape, dtype=np.uint8)
        arr[g] = 1
        arr[l] = 2
        with rasterio.open(tif, "w", **profile) as dst:
            dst.write(arr, 1)

    summary = {}
    keep = {}
    apr_shifted = None
    for label, apr_use in (("grid", apr_aligned), ("shiftcorr", None)):
        if apr_use is None:
            apr_use = shift_prob(apr_aligned, PC_DX, PC_DY)
            apr_shifted = apr_use
        valid = (jan_p > 0.0) & (apr_use > 0.0)
        for key, masks in build_masks(jan_p, apr_use, valid).items():
            name = f"{key}_{label}"
            write_mask(masks["gain"], OUT / f"{name}_gain.tif", aligned_path)
            write_mask(masks["loss"], OUT / f"{name}_loss.tif", aligned_path)
            summary[name] = summarize(masks, name)
            if name == "A_grid":
                keep[name] = masks
            del masks
        if apr_shifted is not None:
            del apr_shifted
            apr_shifted = None

    # C: object-filtered, applied to B0.5 and B0.7 (both alignment variants)
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
            if name in ("C0.5_grid", "C0.7_grid", "C0.7_shiftcorr"):
                keep[name] = {"gain": cg, "loss": cl}
            del g0, l0, cg, cl

    # combined change masks for headline variants
    for key in ("A_grid", "B0.5_grid", "B0.7_grid", "B0.8_grid", "C0.5_grid", "C0.7_grid", "C0.7_shiftcorr"):
        if key not in summary:
            continue
        if key in keep:
            g, l = keep[key]["gain"], keep[key]["loss"]
        else:
            with rasterio.open(OUT / f"{key}_gain.tif") as src:
                g = src.read(1) == 1
            with rasterio.open(OUT / f"{key}_loss.tif") as src:
                l = src.read(1) == 1
        write_change(g, l, OUT / f"{key}_change.tif", aligned_path)
        if key not in keep:
            del g, l
    (REPORTS / "old_change_baseline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- figures ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(JAN_PROB) as src:
        jan_gray = src.read(
            1,
            out_shape=(1, src.height // 4, src.width // 4),
            resampling=Resampling.average,
        ).astype(np.float32)
    pos = jan_gray[jan_gray > 0]
    jan_gray = np.clip((jan_gray - pos.mean()) / (pos.std() * 3 + 1e-6), 0, 1)

    def show_change(ax, jan_g, g, l, title, subsample=True):
        if subsample:
            g = g[::4, ::4]
            l = l[::4, ::4]
        rgb = np.stack([jan_g, jan_g, jan_g], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        rgb[..., 0] = np.maximum(rgb[..., 0], g.astype(np.float32))
        rgb[..., 2] = np.maximum(rgb[..., 2], l.astype(np.float32))
        ax.imshow(rgb)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    rows = ["A_grid", "B0.7_grid", "C0.7_grid", "C0.7_shiftcorr"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, key in zip(axes.ravel(), rows):
        if key in keep:
            g, l = keep[key]["gain"], keep[key]["loss"]
        else:
            with rasterio.open(OUT / f"{key}_change.tif") as src:
                ch = src.read(1)
            g, l = ch == 1, ch == 2
            del ch
        show_change(ax, jan_gray, g, l, key)
        if key not in keep:
            del g, l
    fig.suptitle("Old SegFormer difference baselines (red=gain, blue=loss)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_overview.png", dpi=110)
    plt.close(fig)

    # zoom crops
    crops = [((15000, 4000), "crop_north"), ((12000, 24000), "crop_east"), ((18000, 12000), "crop_center")]
    for (r0, c0), name in crops:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        with rasterio.open(JAN_PROB) as src:
            jg = src.read(1, window=rasterio.windows.Window(c0, r0, 1024, 1024))
        jg = np.clip((jg - jg[jg > 0].mean()) / (jg[jg > 0].std() * 3 + 1e-6), 0, 1)
        axes[0].imshow(jg, cmap="gray")
        axes[0].set_title("January prob")
        g = keep["A_grid"]["gain"][r0 : r0 + 1024, c0 : c0 + 1024]
        l = keep["A_grid"]["loss"][r0 : r0 + 1024, c0 : c0 + 1024]
        show_change(axes[1], jg, g, l, "A raw XOR", subsample=False)
        g = keep["C0.7_grid"]["gain"][r0 : r0 + 1024, c0 : c0 + 1024]
        l = keep["C0.7_grid"]["loss"][r0 : r0 + 1024, c0 : c0 + 1024]
        show_change(axes[2], jg, g, l, "C0.7 object-filtered", subsample=False)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=110)
        plt.close(fig)

    # object size distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = ["C0.5_grid", "C0.7_grid", "C0.7_shiftcorr"]
    bins_names = ["tiny_<30m2", "30-100m2", "100-500m2", "500m2-0.5ha", ">=0.5ha"]
    x = np.arange(len(bins_names))
    w = 0.25
    for i, k in enumerate(keys):
        vals = [summary[k]["gain"]["size_bins"][b] + summary[k]["loss"]["size_bins"][b] for b in bins_names]
        ax.bar(x + (i - 1) * w, vals, width=w, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(bins_names, rotation=20, ha="right")
    ax.set_ylabel("components")
    ax.set_title("Change component size distribution (old baseline)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "component_size_distribution.png", dpi=110)
    plt.close(fig)

    # ---- markdown ----
    md = [
        "# Old SegFormer Difference Baseline (Phase D)",
        "",
        f"- 模型：`segformer_b5_zhelin_local_best_20260624.pth`（iter-6000；`predict_month.py`），概率图来自 `outputs/segmentation/2026-01|04/building_probability.tif`。",
        "- April 概率图已重采样到 January 网格（derivative aligned：`outputs/old_change_baseline/april_probability_aligned_jan_grid.tif`），原始影像未修改。",
        "- `shiftcorr` 变体在重采样后再施加配准审计的 PC 校正（April 移动 dx=-1.61, dy=+0.17 Jan px）。",
        "- 像素面积按 0.5 m x 0.5 m = 0.25 m² 计；C 变体对象过滤：去 <30 m² 组件、填 <20 m² 孔、3x3 opening。",
        "- **不计算 Shanghai Change IoU/F1（GT Gate 前禁止）**；以下均为 candidate 统计。",
        "",
        "## 汇总统计",
        "",
        "| Variant | Changed px | Changed area (m²) | Gain objs | Loss objs | Gain area (m²) | Loss area (m²) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, s in summary.items():
        md.append(
            f"| {key} | {s['changed_pixels']:,} | {s['changed_area_m2']:,.0f} | "
            f"{s['gain']['count']:,} | {s['loss']['count']:,} | {s['gain']['area_m2']:,.0f} | {s['loss']['area_m2']:,.0f} |"
        )
    md += [
        "",
        "## 观察",
        "",
        "- A（raw XOR）在 content 区域产生大量碎片化候选变化；B（置信过滤）随阈值上升显著减少碎片。",
        "- C（对象过滤）进一步移除 <30 m² 碎块，剩余对象数/面积见上表。",
        "- `shiftcorr`（施加配准校正后）与 `grid` 对比可量化配准残差对伪变化的影响（Phase K 使用）。",
        "- 这些是 **model-derived change candidates**，不代表真实建设/拆除。",
        "",
        "## 输出",
        "",
        "- `outputs/old_change_baseline/*_change.tif`（1=gain, 2=loss, 255=nodata）",
        "- `outputs/old_change_baseline/*_{gain,loss}.tif`",
        "- `outputs/old_change_baseline/baseline_overview.png`、`crop_*.png`、`component_size_distribution.png`",
        "- `reports/old_change_baseline.json`",
        "",
    ]
    (REPORTS / "old_change_baseline.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/old_change_baseline.md/.json")
    for key in ("A_grid", "B0.7_grid", "C0.7_grid", "C0.7_shiftcorr"):
        s = summary[key]
        print(
            f"{key}: changed={s['changed_area_m2']/1e4:.1f} ha | gain objs={s['gain']['count']} | "
            f"loss objs={s['loss']['count']} | gain={s['gain']['area_m2']/1e4:.1f} ha | loss={s['loss']['area_m2']/1e4:.1f} ha"
        )


if __name__ == "__main__":
    main()
