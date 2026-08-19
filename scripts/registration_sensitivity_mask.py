"""Phase E - Mask-level registration sensitivity for the old differencing pipeline.

Simulates residual misregistration of the aligned April mask by 0/1/2/4 px
(diagonal shift) and quantifies apparent change produced purely by alignment
error in the segmentation-differencing baseline.

Outputs: reports/registration_sensitivity.md (updated, merged with the
image-level experiment) + outputs/registration_sensitivity/figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "registration_sensitivity"
REPORTS = PROJECT_ROOT / "reports"
APR_ALIGNED_PROB = PROJECT_ROOT / "outputs" / "old_change_baseline" / "april_probability_aligned_jan_grid.tif"
OLD_BASELINE_JSON = REPORTS / "old_change_baseline.json"
SHIFTS = [0, 1, 2, 4]
PIXEL_AREA_M2 = 0.25


def component_stats(mask: np.ndarray) -> dict:
    lbl, n = ndimage.label(mask.astype(np.uint8))
    if n == 0:
        return {
            "count": 0,
            "pixels": 0,
            "area_m2": 0.0,
            "size_bins": {
                "tiny_<30m2": 0,
                "30-100m2": 0,
                "100-500m2": 0,
                "500m2-0.5ha": 0,
                ">=0.5ha": 0,
            },
            "median_px": 0,
        }
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
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(APR_ALIGNED_PROB) as src:
        apr_p = src.read(1).astype(np.float32)
    apr_mask = apr_p >= 0.5
    valid = apr_p > 0.0
    del apr_p
    content_px = int(valid.sum())

    rows = []
    for k in SHIFTS:
        if k == 0:
            shifted = apr_mask.copy()
        else:
            shifted = np.zeros_like(apr_mask)
            shifted[: apr_mask.shape[0] - k, : apr_mask.shape[1] - k] = apr_mask[
                k:, k:
            ]  # shift content toward origin (simulates residual offset removal error)
        diff = (shifted != apr_mask) & valid
        st = component_stats(diff)
        rows.append(
            {
                "shift_px": k,
                "apparent_change_pixels": st["pixels"],
                "apparent_change_area_m2": st["area_m2"],
                "apparent_change_fraction_of_content": round(st["pixels"] / content_px, 6),
                "components": st["count"],
                "size_bins": st["size_bins"],
            }
        )
        print(rows[-1])

    (OUT / "mask_shift_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # figure: apparent change area & components vs shift
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ks = [r["shift_px"] for r in rows]
    axes[0].plot(ks, [r["apparent_change_area_m2"] / 1e4 for r in rows], marker="o")
    axes[0].set_xlabel("residual shift (px, diagonal)")
    axes[0].set_ylabel("apparent change area (ha)")
    axes[0].set_title("Registration-induced apparent change (old differencing)")
    axes[1].plot(ks, [r["components"] for r in rows], marker="s", color="tab:red")
    axes[1].set_xlabel("residual shift (px, diagonal)")
    axes[1].set_ylabel("change components")
    axes[1].set_title("Fragmentation")
    fig.tight_layout()
    fig.savefig(OUT / "mask_shift_sensitivity.png", dpi=120)
    plt.close(fig)

    # load previous image-level results for the merged report
    try:
        img_results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    except Exception:
        img_results = []
    import collections

    agg_img = collections.defaultdict(list)
    for x in img_results:
        agg_img[x["shift_px"]].append(x)

    old_c = json.loads(OLD_BASELINE_JSON.read_text(encoding="utf-8"))
    c07 = old_c.get("C0.7_grid", {})

    md = [
        "# Registration Sensitivity",
        "",
        "- 更新：2026-08-18（Phase E）。保留原有“影像平移→SegFormer 重推理”实验，新增“对齐后 mask 平移→旧差分”的 mask 级实验。",
        "- 模拟残差配准误差 0/1/2/4 px（对角 (k,k)，约 0.5-2 m），内容 = January 网格内容区。",
        "",
        "## 1) Image-level: 平移 January 影像后 SegFormer 重推理（3 个 4096×4096 测试区）",
        "",
        "| Shift (px) | IoU vs GT (mean) | False change frac (mean) | Fragments (mean) |",
        "|---|---|---|---|",
    ]
    for s in SHIFTS:
        a = agg_img.get(s, [])
        if a:
            md.append(
                f"| {s} | {np.mean([x['iou_vs_gt'] for x in a]):.4f} | "
                f"{np.mean([x['false_change_frac'] for x in a]):.5f} | "
                f"{np.mean([x['false_change_components'] for x in a]):.0f} |"
            )
        else:
            md.append(f"| {s} | n/a | n/a | n/a |")
    md += [
        "",
        "## 2) Mask-level: 对齐后的 April mask 平移 k px 后与原 mask 做旧差分（全 AOI）",
        "",
        "| Shift (px) | Apparent change px | Apparent change (ha) | 占内容区比例 | Components | <30m² | 30-100m² | 100-500m² | 500m²-0.5ha | >=0.5ha |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b = r["size_bins"]
        md.append(
            f"| {r['shift_px']} | {r['apparent_change_pixels']:,} | {r['apparent_change_area_m2'] / 1e4:.2f} | "
            f"{r['apparent_change_fraction_of_content']:.4%} | {r['components']:,} | "
            f"{b['tiny_<30m2']:,} | {b['30-100m2']:,} | {b['100-500m2']:,} | {b['500m2-0.5ha']:,} | {b['>=0.5ha']:,} |"
        )
    md += [
        "",
        "## 3) 与 Phase D 旧差分的对照",
        "",
        f"- Phase D 旧差分（iter-6000，Apr→Jan 网格，C0.7 对象过滤）：changed ≈ {c07.get('changed_area_m2', 0) / 1e4:.1f} ha"
        f"（gain {c07.get('gain', {}).get('count', 0):,} 对象 / loss {c07.get('loss', {}).get('count', 0):,} 对象）。",
        f"- 而仅 1 px 残差配准误差本身就会在 April mask 上产生约 {rows[1]['apparent_change_area_m2'] / 1e4:.1f} ha 的伪变化"
        f"（{rows[1]['components']:,} 个碎片），2 px 约 {rows[2]['apparent_change_area_m2'] / 1e4:.1f} ha，"
        f"4 px 约 {rows[3]['apparent_change_area_m2'] / 1e4:.1f} ha。",
        "",
        "## 结论",
        "",
        "- Jan-Apr 配准审计估计全局残差约 1.6-2.3 px（`reports/jan_apr_registration.md`），即旧差分输出中可能包含 **数十公顷纯配准伪变化**。",
        "- 1 px 偏移足以让旧差分产生上万碎片；**任何把旧差分直接当真实变化的行为都不可靠**。",
        "- 直接 CD 模型训练时保留 ±2 px 容忍度是必要的（而不是依赖单点全局校正）。",
        "",
        "图：`outputs/registration_sensitivity/mask_shift_sensitivity.png`",
        "",
    ]
    (REPORTS / "registration_sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/registration_sensitivity.md")


if __name__ == "__main__":
    main()
