"""Phase K - structural comparison of Old SegFormer-differencing vs Direct CD.

Inputs:
  - old baseline: outputs/old_change_baseline/{A_grid,C0.7_grid,C0.7_shiftcorr}_change.tif
  - direct CD:    outputs/shanghai_zero_shot/<tag>/change_mask.tif (+ probability)
Outputs:
  - outputs/old_vs_direct/ figures + stats.json
  - outputs/failure_analysis/ disagreement gallery patches
  - reports/pre_gt_failure_analysis.md

No Shanghai quantitative accuracy metrics (GT Gate).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "old_vs_direct"
FAIL = PROJECT_ROOT / "outputs" / "failure_analysis"
REPORTS = PROJECT_ROOT / "reports"
OLD_DIR = PROJECT_ROOT / "outputs" / "old_change_baseline"
JAN_IMG = r"<LOCAL_SHANGHAI_DATA>/2026-01.tif"
APR_ALIGNED = OLD_DIR / "april_probability_aligned_jan_grid.tif"


def load_change(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        ch = src.read(1)
    valid = ch != 255
    gain = (ch == 1) & valid
    loss = (ch == 2) & valid
    return gain, loss


def comp_stats(mask: np.ndarray) -> dict:
    lbl, n = ndimage.label(mask.astype(np.uint8))
    if n == 0:
        return {"count": 0, "pixels": int(mask.sum()), "area_ha": 0.0}
    ids, counts = np.unique(lbl[mask], return_counts=True)
    sizes = counts.astype(np.int64)
    del lbl
    return {
        "count": int(n),
        "pixels": int(mask.sum()),
        "area_ha": round(float(mask.sum() * 0.25 / 1e4), 2),
        "median_px": int(np.median(sizes)),
        "max_px": int(sizes.max()),
        "size_bins": {
            "tiny_<30m2": int((sizes < 120).sum()),
            "30-100m2": int(((sizes >= 120) & (sizes < 400)).sum()),
            "100-500m2": int(((sizes >= 400) & (sizes < 2000)).sum()),
            "500m2-0.5ha": int(((sizes >= 2000) & (sizes < 20000)).sum()),
            ">=0.5ha": int((sizes >= 20000).sum()),
        },
    }


def boundary_concentration(change: np.ndarray, bmask: np.ndarray, r: int = 3) -> dict:
    """Fraction of change pixels within r px of any building-mask boundary."""
    b = bmask > 0
    bd = ndimage.binary_dilation(b, iterations=r) ^ ndimage.binary_erosion(b, iterations=r)
    total = int(change.sum())
    near = int((change & bd).sum())
    return {
        "change_pixels": total,
        "within_boundary_pixels": near,
        "boundary_share": round(near / max(total, 1), 5),
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    FAIL.mkdir(parents=True, exist_ok=True)

    old_keys = ["A_grid", "C0.7_grid", "C0.7_shiftcorr"]
    direct_tags = [p.name for p in (PROJECT_ROOT / "outputs" / "shanghai_zero_shot").iterdir() if (p / "change_mask.tif").exists()]
    if not direct_tags:
        print("No zero-shot outputs found yet. Run opencd_zero_shot_shanghai.py first.")
        return

    # building mask (union of Jan/Apr probabilities on Jan grid)
    with rasterio.open(APR_ALIGNED) as src:
        apr_p = src.read(1)
    with rasterio.open(PROJECT_ROOT / "outputs" / "segmentation" / "2026-01" / "building_probability.tif") as src:
        jan_p = src.read(1)
    bmask = (jan_p >= 0.5) | (apr_p >= 0.5)
    valid = (jan_p > 0) & (apr_p > 0)
    del jan_p, apr_p

    methods = {}
    for k in old_keys:
        g, l = load_change(OLD_DIR / f"{k}_change.tif")
        methods[k] = {"gain": g, "loss": l, "change": (g | l) & valid, "kind": "old"}
    for tag in direct_tags:
        g, l = load_change(PROJECT_ROOT / "outputs" / "shanghai_zero_shot" / tag / "change_mask.tif")
        methods[tag] = {"gain": g, "loss": l, "change": (g | l) & valid, "kind": "direct"}

    stats = {}
    for name, m in methods.items():
        stats[name] = {
            "kind": m["kind"],
            "change": comp_stats(m["change"]),
            "gain": comp_stats(m["gain"]),
            "loss": comp_stats(m["loss"]),
            "boundary": boundary_concentration(m["change"], bmask),
            "change_fraction_of_content": round(float(m["change"].sum() / max(int(valid.sum()), 1)), 6),
        }

    # pairwise overlap (direct vs old C0.7)
    overlap = {}
    direct = [n for n in methods if methods[n]["kind"] == "direct"]
    oldc = methods["C0.7_grid"]["change"]
    for d in direct:
        inter = int((methods[d]["change"] & oldc).sum())
        union = int((methods[d]["change"] | oldc).sum())
        overlap[d] = {
            "iou_with_C0.7": round(inter / max(union, 1), 4),
            "old_change_covered_by_direct": round(inter / max(int(oldc.sum()), 1), 4),
            "direct_change_covered_by_old": round(inter / max(int(methods[d]["change"].sum()), 1), 4),
        }

    (OUT / "stats.json").write_text(
        json.dumps({"methods": stats, "overlap_with_C0.7": overlap}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # figures: overview (decimated)
    def show(ax, ch, title):
        ch = ch[::4, ::4]
        rgb = np.stack([ch, ch, ch], axis=-1).astype(np.float32) * 0
        g = (ch == 1)
        l = (ch == 2)
        rgb[..., 0] = np.maximum(rgb[..., 0], g.astype(np.float32))
        rgb[..., 2] = np.maximum(rgb[..., 2], l.astype(np.float32))
        ax.imshow(rgb)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    rows = ["A_grid", "C0.7_grid"] + direct
    fig, axes = plt.subplots(2, max(2, (len(rows) + 1) // 2), figsize=(16, 9))
    for ax, name in zip(axes.ravel(), rows):
        ch = np.zeros_like(valid)
        ch[methods[name]["gain"]] = 1
        ch[methods[name]["loss"]] = 2
        ch[~valid] = 255
        show(ax, ch, name)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("Old differencing vs Direct CD (red=gain, blue=loss)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=110)
    plt.close(fig)

    # failure gallery: sample disagreement patches
    direct_name = direct[0]
    old_ch = methods["C0.7_grid"]["change"]
    direct_ch = methods[direct_name]["change"]
    only_direct = direct_ch & ~old_ch
    only_old = old_ch & ~direct_ch
    rng = np.random.default_rng(2026)

    def sample_patches(mask, n, tag):
        lbl, nlab = ndimage.label(mask.astype(np.uint8))
        if nlab == 0:
            return
        ids, counts = np.unique(lbl[mask], return_counts=True)
        order = np.argsort(counts)[::-1]
        picked = []
        for idx in order:
            i = int(ids[idx])
            ys, xs = np.nonzero(lbl == i)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            h, w = mask.shape
            r0, c0 = max(0, cy - 256), max(0, cx - 256)
            r0 = min(r0, h - 512)
            c0 = min(c0, w - 512)
            picked.append((r0, c0))
            if len(picked) >= n:
                break
        apr_rgb_path = PROJECT_ROOT / "outputs" / "shanghai_zero_shot" / direct[0] / "april_aligned_jan_grid.tif"
        with rasterio.open(JAN_IMG) as j, rasterio.open(apr_rgb_path) as a:
            for i, (r0, c0) in enumerate(picked):
                jan = j.read([1, 2, 3], window=rasterio.windows.Window(c0, r0, 512, 512))
                apr = a.read([1, 2, 3], window=rasterio.windows.Window(c0, r0, 512, 512))
                fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                axes[0].imshow(np.moveaxis(jan, 0, -1))
                axes[0].set_title("Jan")
                axes[1].imshow(np.moveaxis(apr, 0, -1))
                axes[1].set_title("Apr (aligned)")
                for ax, (nm, m) in zip(axes[2:], [("old C0.7", old_ch), (direct_name, direct_ch)]):
                    win = m[r0 : r0 + 512, c0 : c0 + 512]
                    rgb = np.zeros((512, 512, 3), dtype=np.float32)
                    rgb[..., 0] = np.clip(win.astype(np.float32), 0, 1)
                    rgb[..., 2] = win.astype(np.float32)
                    ax.imshow(rgb)
                    ax.set_title(nm)
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(f"{tag} patch {i} at (r={r0}, c={c0})", fontsize=10)
                fig.tight_layout()
                fig.savefig(FAIL / f"{tag}_{i:02d}_r{r0}_c{c0}.png", dpi=100)
                plt.close(fig)

    sample_patches(only_direct, 6, "direct_only")
    sample_patches(only_old, 6, "old_only")

    md = [
        "# Pre-GT Failure Analysis: Old Differencing vs Direct CD",
        "",
        f"- 生成时间：2026-08-18（Phase K）。比较对象：Old = SegFormer iter-6000 差分（A raw / C0.7 对象过滤 / C0.7+配准校正），Direct = {', '.join(direct)}（WHU 训练 zero-shot）。",
        "- **不计算 Shanghai 指标**（GT Gate 前禁止）；以下全部为结构/定性统计。",
        "",
        "## 结构统计",
        "",
        "| Method | Change area (ha) | Components | <30m² | 30-100m² | 100-500m² | 500m²-0.5ha | >=0.5ha | boundary share |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        c = s["change"]
        b = c["size_bins"]
        md.append(
            f"| {name} | {c['area_ha']:.2f} | {c['count']:,} | {b['tiny_<30m2']:,} | {b['30-100m2']:,} | "
            f"{b['100-500m2']:,} | {b['500m2-0.5ha']:,} | {b['>=0.5ha']:,} | {s['boundary']['boundary_share']:.1%} |"
        )
    md += [
        "",
        "## 与 C0.7 旧差分的重叠（direct 方法）",
        "",
        "| Direct method | IoU with C0.7 | old covered by direct | direct covered by old |",
        "|---|---|---|---|",
    ]
    for d, o in overlap.items():
        md.append(f"| {d} | {o['iou_with_C0.7']:.4f} | {o['old_change_covered_by_direct']:.1%} | {o['direct_change_covered_by_old']:.1%} |")
    md += [
        "",
        "## 观察（候选结论，待 GT Gate 后验证）",
        "",
        "- 直接 CD 的碎片化程度通常低于旧差分（见 components 与 <30m² 占比）。",
        "- boundary share 反映变化是否集中在建筑边缘（配准伪变化特征）；直接 CD 若显著更低，说明其边缘伪变化抑制更好。",
        "- 重叠低且方向一致的区域是可能真实变化；`direct_only` / `old_only` gallery 见 `outputs/failure_analysis/`。",
        "",
    ]
    (REPORTS / "pre_gt_failure_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"stats": stats, "overlap": overlap}, ensure_ascii=False, indent=2))
    print("wrote reports/pre_gt_failure_analysis.md")


if __name__ == "__main__":
    main()
