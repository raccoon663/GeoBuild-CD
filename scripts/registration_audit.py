"""Registration audit: compare every month against the January reference.

Outputs:
  - reports/registration_audit.md
  - reports/registration_audit.json
  - outputs/registration/<month>_*.png  (overlay / edges / difference)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REF = "2026-01"
MONTHS = {
    "2026-01": r"<LOCAL_SHANGHAI_DATA>/2026-01.tif",
    "2026-02": r"<LOCAL_SHANGHAI_DATA>/2026-02.tif",
    "2026-03": r"<LOCAL_SHANGHAI_DATA>/2026-03.tif",
    "2026-04": r"<LOCAL_SHANGHAI_DATA>/2026-04.tif",
    "2026-05-31": r"<LOCAL_SHANGHAI_DATA>/2026-05-31.tif",
}

CONTENT = json.loads(
    (PROJECT_ROOT / "reports" / "content_extents.json").read_text(encoding="utf-8")
)

DECIMATION = 10


def read_window(
    path: str,
    bounds,
    out_shape=None,
    dec: int = DECIMATION,
    max_bands: int = 3,
    round_win: bool = True,
) -> np.ndarray:
    """Read a geographic bounds window as float32 [bands,h,w].

    If out_shape is given the data are resampled to that shape, which
    aligns the target grid with the reference grid.
    """
    with rasterio.open(path) as src:
        n_bands = min(src.count, max_bands)
        if out_shape is None:
            win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            if round_win:
                win = win.round_offsets().round_lengths()
            h, w = int(win.height), int(win.width)
            out_h, out_w = max(h // dec, 1), max(w // dec, 1)
            out_shape = (n_bands, out_h, out_w)
            data = src.read(
                indexes=list(range(1, n_bands + 1)),
                window=win,
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)
            return data
        win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        data = src.read(
            indexes=list(range(1, n_bands + 1)),
            window=win,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.average,
        ).astype(np.float32)
        return data


def gray(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 1:
        return data[0]
    return data[:3].mean(axis=0)


def content_bounds(month: str):
    b = CONTENT[month]["content_bounds"]
    if b is None:
        raise RuntimeError(f"no content for {month}")
    return b


def main():
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = PROJECT_ROOT / "outputs" / "registration"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_b = content_bounds(REF)
    ref_gray = None
    results = {}

    for month, path in MONTHS.items():
        if month == REF:
            continue
        tb = content_bounds(month)
        common = [
            max(ref_b[0], tb[0]),
            max(ref_b[1], tb[1]),
            min(ref_b[2], tb[2]),
            min(ref_b[3], tb[3]),
        ]
        if common[2] <= common[0] or common[3] <= common[1]:
            results[month] = {"error": "no overlap with reference content"}
            continue
        # reference window defines the decimated grid
        with rasterio.open(MONTHS[REF]) as src:
            win_ref = rasterio.windows.from_bounds(*common, transform=src.transform)
            win_ref = win_ref.round_offsets().round_lengths()
            out_h, out_w = max(int(win_ref.height) // DECIMATION, 1), max(int(win_ref.width) // DECIMATION, 1)
        ref_arr = read_window(MONTHS[REF], common, out_shape=(3, out_h, out_w))
        tgt_arr = read_window(path, common, out_shape=(3, out_h, out_w))
        ref_g = gray(ref_arr)
        tgt_g = gray(tgt_arr)
        assert ref_g.shape == tgt_g.shape, (ref_g.shape, tgt_g.shape)
        if ref_gray is None:
            ref_gray = ref_g
        h, w = ref_g.shape

        # ECC translation estimate (robust to illumination)
        warp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        ref8 = np.clip(ref_g / 255.0 * 255, 0, 255).astype(np.uint8)
        tgt8 = np.clip(tgt_g / 255.0 * 255, 0, 255).astype(np.uint8)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
            (cc, warp_ecc) = cv2.findTransformECC(
                ref8, tgt8, warp, cv2.MOTION_TRANSLATION, criteria, None, 3
            )
            ecc_dx = float(warp_ecc[0, 2])
            ecc_dy = float(warp_ecc[1, 2])
        except Exception as exc:
            ecc_dx, ecc_dy, cc = None, None, None

        # Phase correlation (sub-pixel, robust fallback / cross-check)
        from skimage.registration import phase_cross_correlation

        try:
            shift, error, diffphase = phase_cross_correlation(
                ref_g, tgt_g, upsample_factor=10
            )
            pc_dy, pc_dx = float(shift[0]) * DECIMATION, float(shift[1]) * DECIMATION
        except Exception:
            pc_dx, pc_dy = float("nan"), float("nan")

        result = {
            "common_bounds": [round(v, 7) for v in common],
            "common_decimated_shape": [h, w],
            "ecc_shift_px_native": (
                {"dx": round(ecc_dx * DECIMATION, 2), "dy": round(ecc_dy * DECIMATION, 2), "cc": round(float(cc), 4)}
                if ecc_dx is not None
                else None
            ),
            "phase_corr_shift_px_native": {"dx": round(pc_dx, 2), "dy": round(pc_dy, 2)},
        }
        results[month] = result
        print(month, "ECC shift px:", result["ecc_shift_px_native"], "PC:", result["phase_corr_shift_px_native"])

        # ---------- visuals ----------
        # stretch both
        def stretch(g):
            g2 = g.copy()
            lo, hi = np.percentile(g2[g2 > 5], (2, 98)) if (g2 > 5).any() else (0, 255)
            return np.clip((g2 - lo) / max(hi - lo, 1e-6), 0, 1)

        r_s = stretch(ref_g)
        t_s = stretch(tgt_g)
        rgb = np.zeros((h, w, 3))
        rgb[:, :, 0] = r_s
        rgb[:, :, 2] = t_s
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(rgb)
        ax.set_title(f"{REF} (red) vs {month} (cyan)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{month}_rgb_overlay.png", dpi=100)
        plt.close(fig)

        # edge overlay
        edges_r = cv2.Canny(ref8, 40, 120)
        edges_t = cv2.Canny(tgt8, 40, 120)
        edge_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        edge_rgb[edges_r > 0] = (255, 0, 0)
        edge_rgb[edges_t > 0] = (0, 255, 255)
        edge_rgb[(edges_r > 0) & (edges_t > 0)] = (255, 255, 0)
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(edge_rgb)
        ax.set_title(f"{REF} (red) vs {month} (cyan) edges")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{month}_edge_overlay.png", dpi=100)
        plt.close(fig)

        # difference visualization (raw, no shift applied)
        diff = np.abs(r_s - t_s)
        fig, ax = plt.subplots(figsize=(11, 8))
        im = ax.imshow(diff, cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{REF} vs {month} raw intensity difference")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035)
        fig.tight_layout()
        fig.savefig(out_dir / f"{month}_difference.png", dpi=100)
        plt.close(fig)

    (PROJECT_ROOT / "reports" / "registration_audit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- markdown ----------------
    import rasterio as rio

    lines = ["# Registration Audit", ""]
    lines.append(f"Reference month: `{REF}`  (`{MONTHS[REF]}`)")
    lines.append("")
    lines.append("## Grid metadata comparison")
    lines.append("")
    lines.append("| Month | Width | Height | CRS | Pixel size (deg) | Full bounds (L,B,R,T) |")
    lines.append("|---|---|---|---|---|---|")
    with rio.open(MONTHS[REF]) as src:
        ref_bounds = [round(v, 7) for v in src.bounds]
    for month, path in MONTHS.items():
        with rio.open(path) as src:
            b = [round(v, 7) for v in src.bounds]
            lines.append(
                f"| {month} | {src.width} | {src.height} | {src.crs} | {abs(src.transform.a):.2e} | "
                f"{b[0]}, {b[1]}, {b[2]}, {b[3]} |"
            )
    lines.append("")
    lines.append("## Content (non-black) coverage comparison")
    lines.append("")
    for month, path in MONTHS.items():
        cb = content_bounds(month)
        lines.append(
            f"- {month}: content bounds `{cb}`, fraction {CONTENT[month]['content_fraction']:.3f}"
        )
    lines.append("")
    lines.append("## Estimated shifts vs reference (native 0.5 m pixels)")
    lines.append("")
    lines.append("| Month | ECC dx (px) | ECC dy (px) | Phase-correlation dx | Phase-correlation dy |")
    lines.append("|---|---|---|---|---|")
    for month, r in results.items():
        if "error" in r:
            lines.append(f"| {month} | {r['error']} | | | |")
            continue
        e = r["ecc_shift_px_native"]
        if e:
            lines.append(f"| {month} | {e['dx']} | {e['dy']} | {r['phase_corr_shift_px_native']['dx']} | {r['phase_corr_shift_px_native']['dy']} |")
        else:
            lines.append(f"| {month} | n/a (ECC failed) | | {r['phase_corr_shift_px_native']['dx']} | {r['phase_corr_shift_px_native']['dy']} |")
    lines.append("")
    lines.append("## Interpretation / risk")
    lines.append("")
    lines.append("- All months share CRS EPSG:4326 and the same nominal 5e-6 deg pixel size, but the **raster canvases have different extents and grid origins**. Pixel grids are therefore **not** aligned across months.")
    lines.append("- ECC estimates a global translation on a decimated common-area crop. Small values (< ~2 px) indicate acceptable coarse registration; larger values indicate a systematic offset that must be handled before any pixel-wise comparison.")
    lines.append("- Seasonal appearance, shadow, vegetation and illumination differences dominate raw intensity differences; raw difference maps must **not** be interpreted as building change.")
    lines.append("- The estimated shift is a global approximation. Local residual offsets (building parallax, orthorectification) can still exist.")
    lines.append("")
    lines.append("Visualizations are in `outputs/registration/`.")
    (PROJECT_ROOT / "reports" / "registration_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Wrote reports/registration_audit.md")


if __name__ == "__main__":
    main()
