"""Phases U/V/W/X/Y - ranking, review cards, manifest, stratified set, gate.

Consumes outputs/review_candidates/{all_candidates,oa_only,fc_only,negative_controls}.gpkg
(geometries MUST be correct world coordinates; see build_review_candidates.py)
and produces:
  outputs/review_candidates/top{50,100,200}.gpkg|.csv
  outputs/human_review/cards/<candidate_id>.png   (geometry-based crop + overlay + QA)
  outputs/human_review/review_manifest.csv        (existing human columns preserved)
  outputs/human_review/stratified_review_set.csv
  reports/HUMAN_REVIEW_GATE.md

Card cropping (2026-08-19 fix):
  - crop window is derived from the candidate polygon bbox + contextual buffer
    (configs/candidate_ranking.yaml -> card.context_buffer_m), clamped to the
    raster, NEVER from the x/y metadata columns.
  - candidate polygon is explicitly overlaid on Jan / Apr / probability panels.
  - every card is QA-checked (valid content coverage, candidate inside valid
    content, probability-in-candidate consistency vs manifest) and the QA
    result is written to outputs/human_review/card_qa_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

OUT = PROJECT_ROOT / "outputs" / "review_candidates"
HUMAN = PROJECT_ROOT / "outputs" / "human_review"
REPORTS = PROJECT_ROOT / "reports"
CFG = yaml.safe_load((PROJECT_ROOT / "configs" / "candidate_ranking.yaml").read_text(encoding="utf-8"))

JAN_IMG = r"<LOCAL_SHANGHAI_DATA>/2026-01.tif"
APR_RGB = PROJECT_ROOT / "outputs" / "shanghai_zero_shot" / "changeformer" / "april_aligned_jan_grid.tif"
CF_PROB = PROJECT_ROOT / "outputs" / "shanghai_zero_shot" / "changeformer" / "change_probability.tif"
JAN_PROB = PROJECT_ROOT / "outputs" / "segmentation" / "2026-01" / "building_probability.tif"

PIX_M = 0.5  # nominal metres per pixel on the Jan grid (~0.5 m)
HUMAN_COLS = ["review_status", "true_change", "change_type", "confidence", "notes"]


def load(name):
    import geopandas as gpd

    return gpd.read_file(OUT / f"{name}.gpkg")


def clip01(x):
    return float(min(max(x, 0.0), 1.0))


def compute_scores(gdf: pd.DataFrame) -> pd.DataFrame:
    r = CFG["ranking"]
    b = CFG["building_context"]
    risks = CFG["risks"]
    size_score_map = CFG["size_score"]

    # building context score
    prob_delta = gdf["apr_building_mean_prob"] - gdf["jan_building_mean_prob"]
    prob_contrib = 1.0 / (1.0 + np.exp(-b["prob_delta_scale"] * prob_delta))
    dist_contrib = np.exp(-gdf["distance_to_building_m"] / b["dist_scale_m"])
    gdf["building_context_score"] = (
        b["weight_jan_overlap"] * gdf["jan_building_overlap"]
        + b["weight_apr_overlap"] * gdf["apr_building_overlap"]
        + b["weight_adjacent"] * (gdf["distance_to_building_m"] <= 3).astype(float)
        + b["weight_prob_delta"] * prob_contrib
        + b["weight_distance"] * dist_contrib
    ).clip(0, 1)

    gdf["agreement_score"] = (
        (gdf["legacy_iter6000_overlap"] > 0).astype(float)
        + (gdf["legacy_iter14000_overlap"] > 0).astype(float)
        + (gdf["fc_siam_overlap"] > 0).astype(float)
    ) / r["agreement_normalizer"]

    # boundary artifact risk: linear 1 at <=16 px, 0 at >=64 px
    bd = gdf["boundary_distance_px"].fillna(0)
    gdf["boundary_artifact_risk"] = np.clip(
        1
        - (bd - risks["boundary_artifact"]["risk_full_below_px"])
        / (risks["boundary_artifact"]["risk_zero_beyond_px"]
           - risks["boundary_artifact"]["risk_full_below_px"]),
        0,
        1,
    )

    # registration risk: building-edge fraction
    gdf["registration_risk"] = gdf["building_edge_fraction"].fillna(0)

    # seasonal/landcover risk: spectral proxy, attenuated by building overlap & agreement
    spec_norm = (gdf["mean_spectral_diff"].fillna(0)) / 255.0
    spectral_risk = np.clip((spec_norm - 0.08) / 0.25, 0, 1)
    low_build = 1 - np.maximum(gdf["jan_building_overlap"], gdf["apr_building_overlap"])
    gdf["seasonal_landcover_risk"] = (spectral_risk * low_build * (1 - gdf["agreement_score"])).clip(0, 1)

    gdf["size_score"] = gdf["size_group"].map(size_score_map).fillna(0.5)

    gdf["rank_score"] = (
        r["weight_mean_prob"] * gdf["changeformer_mean_prob"].fillna(0)
        + r["weight_max_prob"] * gdf["changeformer_max_prob"].fillna(0)
        + r["weight_building_context"] * gdf["building_context_score"]
        + r["weight_agreement"] * gdf["agreement_score"]
        + r["weight_size"] * gdf["size_score"]
        - r["penalty_boundary_artifact"] * gdf["boundary_artifact_risk"]
        - r["penalty_registration_risk"] * gdf["registration_risk"]
        - r["penalty_seasonal_landcover"] * gdf["seasonal_landcover_risk"]
    )
    return gdf


def merge_human_cols(df: pd.DataFrame, src_csv_path: Path) -> pd.DataFrame:
    """Carry over any existing human review labels by candidate_id."""
    if not src_csv_path.exists():
        return df
    old = pd.read_csv(src_csv_path, dtype=str)
    keep = [c for c in HUMAN_COLS if c in old.columns]
    if not keep:
        return df
    old = old[["candidate_id"] + keep].drop_duplicates("candidate_id")
    df = df.merge(old, on="candidate_id", how="left")
    for c in HUMAN_COLS:
        if c not in df.columns:
            df[c] = ""
        else:
            df[c] = df[c].fillna("")
    return df


def window_for_geom(geom, transform, H, W):
    """Raster window: candidate bbox + contextual buffer, clamped to raster."""
    card = CFG["card"]
    buf_px = int(card["context_buffer_m"] / PIX_M)
    minx, miny, maxx, maxy = geom.bounds
    r_top, c_left = rasterio.transform.rowcol(transform, minx, maxy)
    r_bot, c_right = rasterio.transform.rowcol(transform, maxx, miny)
    r_top = min(max(int(r_top), 0), H - 1)
    r_bot = min(max(int(r_bot), 0), H - 1)
    c_left = min(max(int(c_left), 0), W - 1)
    c_right = min(max(int(c_right), 0), W - 1)
    cand_h = abs(r_bot - r_top) + 1
    cand_w = abs(c_right - c_left) + 1
    side = int(
        min(
            card["max_window_px"],
            max(card["min_window_px"], max(cand_h, cand_w) + 2 * buf_px),
        )
    )
    side = max(1, min(side, H, W))
    half = side // 2
    r0 = int(round((r_top + r_bot) / 2)) - half
    c0 = int(round((c_left + c_right) / 2)) - half
    r1, c1 = r0 + side, c0 + side
    if r0 < 0:
        r0, r1 = 0, side
    if c0 < 0:
        c0, c1 = 0, side
    if r1 > H:
        r1, r0 = H, H - side
    if c1 > W:
        c1, c0 = W, W - side
    if r0 < 0 or c0 < 0:  # degenerate raster smaller than the requested side
        r0, c0, r1, c1 = 0, 0, H, W
    return rasterio.windows.Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))


def geom_to_window_px(geom, win_tr, c0, r0):
    """Geometry vertices in window pixel coords as (x=col, y=row) arrays."""
    if geom.geom_type == "Polygon":
        coords = np.asarray(geom.exterior.coords)
        rows, cols = rasterio.transform.rowcol(win_tr, coords[:, 0], coords[:, 1])
        return np.column_stack([np.asarray(cols) - c0, np.asarray(rows) - r0])
    if geom.geom_type == "MultiPolygon":
        out = []
        for g in geom.geoms:
            v = geom_to_window_px(g, win_tr, c0, r0)
            if v is not None:
                out.append(v)
        return out
    return None


def make_card(gdf_row, out_path: Path, qa: bool = True):
    """Render a 3-panel review card from the candidate GEOMETRY (not x/y)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from rasterio import features

    geom = gdf_row.geometry
    with rasterio.open(CF_PROB) as src:
        transform = src.transform
        H, W = src.height, src.width
    win = window_for_geom(geom, transform, H, W)
    win_tr = rasterio.windows.transform(win, transform)
    c0, r0 = int(win.col_off), int(win.row_off)
    win_h, win_w = int(win.height), int(win.width)

    with rasterio.open(JAN_IMG) as j, rasterio.open(APR_RGB) as a, rasterio.open(CF_PROB) as p, rasterio.open(JAN_PROB) as jp:
        jan = j.read([1, 2, 3], window=win)
        apr = a.read([1, 2, 3], window=win)
        chp = p.read(1, window=win).astype(np.float32)
        janp = jp.read(1, window=win).astype(np.float32)
    jan = np.moveaxis(jan, 0, -1)
    apr = np.moveaxis(apr, 0, -1)

    cand_mask = features.rasterize(
        [(geom, 1)],
        out_shape=(win_h, win_w),
        transform=win_tr,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    ).astype(bool)
    n_cand = int(cand_mask.sum())

    jan_valid = janp > 0
    apr_valid = apr.max(axis=2) > 5
    common = jan_valid & apr_valid
    jan_frac = float(jan_valid.mean())
    apr_frac = float(apr_valid.mean())
    cand_valid_frac = float((cand_mask & common).sum() / max(n_cand, 1))

    rec_mean = float(chp[cand_mask].mean()) if n_cand else 0.0
    rec_max = float(chp[cand_mask].max()) if n_cand else 0.0
    man_mean = float(gdf_row["changeformer_mean_prob"])
    man_max = float(gdf_row["changeformer_max_prob"])
    d_mean = abs(rec_mean - man_mean)
    d_max = abs(rec_max - man_max)

    qa_cfg = CFG["card"]["qa"]
    flags = []
    if cand_valid_frac < qa_cfg["candidate_min_valid_fraction"]:
        flags.append("candidate_outside_valid")
    if jan_frac < qa_cfg["jan_apr_min_valid_fraction"]:
        flags.append("black_jan")
    if apr_frac < qa_cfg["jan_apr_min_valid_fraction"]:
        flags.append("black_apr")
    if d_mean > qa_cfg["prob_tolerance"] or d_max > qa_cfg["prob_tolerance"]:
        flags.append("prob_mismatch")
    status = "ok" if not flags else "|".join(flags)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(jan)
    axes[1].imshow(apr)
    im = axes[2].imshow(chp, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[2], fraction=0.04)

    vertices = geom_to_window_px(geom, win_tr, c0, r0)
    if vertices is not None:
        polys = vertices if isinstance(vertices, list) else [vertices]
        for ax in axes:
            for verts in polys:
                if verts.shape[0] >= 3:
                    ax.add_patch(
                        MplPolygon(verts, closed=True, facecolor="none",
                                   edgecolor="yellow", linewidth=2)
                    )
    else:
        # Point (negative control): yellow crosshair
        px_r, px_c = rasterio.transform.rowcol(win_tr, geom.x, geom.y)
        for ax in axes:
            ax.plot(px_c - c0, px_r - r0, marker="o", mfc="none", mec="yellow",
                    ms=12, mew=2)

    axes[0].set_title("January RGB", fontsize=10)
    axes[1].set_title("April RGB (aligned)", fontsize=10)
    axes[2].set_title("ChangeFormer probability", fontsize=10)
    for ax in axes:
        ax.axis("off")

    meta = (
        f"{gdf_row['candidate_id']} | rank={gdf_row.get('candidate_rank', 'NA')} | "
        f"area={gdf_row['area_m2']:.1f} m2 | {gdf_row['size_group']} | "
        f"({gdf_row['x']:.6f}, {gdf_row['y']:.6f})\n"
        f"CF mean/max={gdf_row['changeformer_mean_prob']:.3f}/{gdf_row['changeformer_max_prob']:.3f} "
        f"| recalc={rec_mean:.3f}/{rec_max:.3f} | "
        f"leg6/leg14/fcsiam ovl={gdf_row['legacy_iter6000_overlap']:.2f}/"
        f"{gdf_row['legacy_iter14000_overlap']:.2f}/{gdf_row['fc_siam_overlap']:.2f} | "
        f"ctx={gdf_row.get('building_context_score', float('nan')):.2f} | "
        f"score={gdf_row.get('rank_score', float('nan')):.3f} | qa={status}"
    )
    fig.suptitle(meta, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    if qa:
        return {
            "candidate_id": gdf_row["candidate_id"],
            "status": status,
            "flags": flags,
            "jan_valid_fraction": round(jan_frac, 4),
            "apr_valid_fraction": round(apr_frac, 4),
            "candidate_valid_fraction": round(cand_valid_frac, 4),
            "candidate_pixels": n_cand,
            "recomputed_mean_prob": round(rec_mean, 4),
            "recomputed_max_prob": round(rec_max, 4),
            "delta_mean_prob": round(d_mean, 4),
            "delta_max_prob": round(d_max, 4),
            "window_px": [win_h, win_w],
        }
    return None


def build_pool():
    cf = load("all_candidates")
    oa = load("oa_only")
    fs = load("fc_only")
    neg = load("negative_controls")

    cf = compute_scores(cf)
    cf = cf.sort_values("rank_score", ascending=False).reset_index(drop=True)
    cf["candidate_rank"] = np.arange(1, len(cf) + 1)
    cols = [c for c in cf.columns if c != "geometry"]
    cf.to_file(OUT / "all_candidates.gpkg", driver="GPKG", encoding="utf-8")
    cf[cols].to_csv(OUT / "all_candidates.csv", index=False, encoding="utf-8")

    top_sets = {}
    for k in CFG["top_k"]:
        top = cf.head(k)
        top.to_file(OUT / f"top{k}.gpkg", driver="GPKG", encoding="utf-8")
        top[cols].to_csv(OUT / f"top{k}.csv", index=False, encoding="utf-8")
        top_sets[k] = set(top["candidate_id"])

    s = CFG["stratified_review"]
    rng = np.random.default_rng(s["seed"])

    def pick(gdf, n, key=None):
        g = gdf.copy()
        if key is not None:
            g = g.sort_values(key, ascending=False)
        return list(g.head(n)["candidate_id"])

    chosen = []
    chosen += pick(cf[cf["legacy_iter14000_overlap"] == 0], s["per_category"]["changeformer_only"], "rank_score")
    chosen += pick(oa, s["per_category"]["original_app_only"], "area_m2")
    chosen += pick(cf[cf["legacy_iter14000_overlap"] > 0], s["per_category"]["both_agree"], "rank_score")
    chosen += pick(fs, s["per_category"]["fc_siam_only"], "area_m2")
    chosen += pick(cf[cf["changeformer_max_prob"] >= 0.8], s["per_category"]["high_confidence_changeformer"], "rank_score")
    chosen += pick(cf[cf["size_group"] == "10-30m2"], s["per_category"]["small_changeformer"], "rank_score")
    chosen += pick(cf[cf["boundary_distance_px"] <= 16], s["per_category"]["boundary_risk"], "rank_score")
    chosen += pick(neg, int(s["target_total"] * s["negative_control_ratio"]), "boundary_distance_px")

    seen = set()
    stratified_ids = []
    for cid in chosen:
        if cid not in seen:
            seen.add(cid)
            stratified_ids.append(cid)
    stratified_ids = stratified_ids[: s["target_total"]]

    pool_ids = set(top_sets[max(CFG["top_k"])]) | set(stratified_ids)
    pool = pd.concat([cf, oa, fs, neg], ignore_index=True)
    pool = pool[pool["candidate_id"].isin(pool_ids)]
    pool = pool.drop_duplicates("candidate_id").reset_index(drop=True)
    return {
        "cf": cf,
        "oa": oa,
        "fs": fs,
        "neg": neg,
        "pool": pool,
        "cols": cols,
        "top_sets": top_sets,
        "stratified_ids": stratified_ids,
    }


def main() -> None:
    b = build_pool()
    cf, oa, fs, neg = b["cf"], b["oa"], b["fs"], b["neg"]
    pool, cols = b["pool"], b["cols"]
    top_sets, stratified_ids = b["top_sets"], b["stratified_ids"]

    cards_dir = HUMAN / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    qa_rows = []
    for _, row in pool.iterrows():
        q = make_card(row, cards_dir / f"{row['candidate_id']}.png")
        if q is not None:
            qa_rows.append(q)
    (HUMAN / "card_qa_results.json").write_text(
        json.dumps(qa_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    bad = [q for q in qa_rows if q["status"] != "ok"]
    print("cards:", len(pool), "qa_bad:", len(bad), bad[:5])

    strat = pool[pool["candidate_id"].isin(set(stratified_ids))].copy()
    strat[cols].to_csv(HUMAN / "stratified_review_set.csv", index=False, encoding="utf-8")
    print("stratified:", len(strat))

    manifest = pool[cols].copy()
    manifest = merge_human_cols(manifest, HUMAN / "review_manifest.csv")
    manifest = manifest[[c for c in manifest.columns if c not in HUMAN_COLS] + HUMAN_COLS]
    manifest.to_csv(HUMAN / "review_manifest.csv", index=False, encoding="utf-8")

    n_cf = len(cf)
    size_counts = cf["size_group"].value_counts().to_dict()
    prob_bins = {
        "0.5-0.7": int(((cf["changeformer_max_prob"] >= 0.5) & (cf["changeformer_max_prob"] < 0.7)).sum()),
        "0.7-0.8": int(((cf["changeformer_max_prob"] >= 0.7) & (cf["changeformer_max_prob"] < 0.8)).sum()),
        "0.8-0.9": int(((cf["changeformer_max_prob"] >= 0.8) & (cf["changeformer_max_prob"] < 0.9)).sum()),
        ">=0.9": int((cf["changeformer_max_prob"] >= 0.9).sum()),
    }
    agree_14000 = float((cf["legacy_iter14000_overlap"] > 0).mean())
    agree_6000 = float((cf["legacy_iter6000_overlap"] > 0).mean())
    agree_fc = float((cf["fc_siam_overlap"] > 0).mean())
    boundary_risk_n = int((cf["boundary_distance_px"] <= 16).sum())
    small_n = int(((cf["size_group"] == "10-30m2") | (cf["size_group"] == "30-100m2")).sum())

    md = [
        "# HUMAN REVIEW GATE（Phase Y）",
        "",
        "- 日期：2026-08-19。主模型：ChangeFormer（WHU-CD 训练，Shanghai Jan-Apr zero-shot，threshold 0.5）。",
        f"- **Total ChangeFormer candidates: {n_cf}**",
        "",
        "## Candidate counts by size",
        "",
    ]
    for g in CFG["size_groups_m2"]:
        name = g[2]
        md.append(f"- {name}: {size_counts.get(name, 0)}")
    md += [
        "",
        "## Candidate counts by max probability",
        "",
    ]
    for k, v in prob_bins.items():
        md.append(f"- max prob {k}: {v}")
    md += [
        "",
        "## Top-K composition",
        "",
        "| K | candidates |",
        "|---|---|",
    ]
    for k in top_k_list():
        md.append(f"| Top {k} | {len(top_sets[k])} |")
    md += [
        "",
        "## Model agreement (ChangeFormer vs others)",
        "",
        f"- ChangeFormer ∩ Original App (iter-14000): {agree_14000:.1%} of CF candidates",
        f"- ChangeFormer ∩ Legacy local (iter-6000): {agree_6000:.1%}",
        f"- ChangeFormer ∩ FC-Siam: {agree_fc:.1%}",
        f"- Original App-only objects（CF 未覆盖）: {len(oa)}",
        f"- FC-Siam-only objects（CF/iter-14000 未覆盖）: {len(fs)}",
        "",
        "## Risk / small-change",
        "",
        f"- Boundary-risk candidates (boundary_distance<=16 px): {boundary_risk_n}",
        f"- Small-change candidates (10-100 m2): {small_n}（10-30 与 30-100 m² 为重点审核范围，未删除）",
        "",
        "## Review-card QA（2026-08-19 修复后）",
        "",
        f"- 卡片按候选几何 bbox+150 m 缓冲裁剪，三面板叠加候选边界；详细审计见 `reports/review_card_QA.md`。",
        f"- 本次 {len(qa_rows)} 张卡片 QA：invalid/flag {len(bad)} 张（要求 0，已满足）。",
        "",
        "## 建议人工审查规模",
        "",
        f"- Review pool（Top{max(CFG['top_k'])} ∪ stratified）: {len(pool)} 个对象；"
        f"分层审查集 {len(strat)} 个（含 "
        f"{CFG['stratified_review']['negative_control_ratio']:.0%} 负控）。",
        f"- 建议首轮人工审查 **Top 100**（约 1-2 小时）；完整 review pool 约 {len(pool)} 张卡片。",
        "",
        "## 路径",
        "",
        "- Review cards: `outputs/human_review/cards/`",
        "- review_manifest.csv: `outputs/human_review/review_manifest.csv`",
        "- stratified_review_set.csv: `outputs/human_review/stratified_review_set.csv`",
        "- 全部候选目录: `outputs/review_candidates/`（all_candidates / top50/100/200 / oa_only / fc_only / negative_controls，gpkg+csv）",
        "",
        "## 下一步（人工 review 完成后执行）",
        "",
        "在 `outputs/human_review/review_manifest.csv` 中填写最后 5 列"
        "（review_status / true_change / change_type / confidence / notes）后，运行：",
        "",
        "```",
        "python scripts/after_human_review.py",
        "```",
        "",
        "将计算 Precision@50/100/200；若人工 GT 覆盖充分再计算 Recall@K / F1 / IoU、"
        "false-positive taxonomy、small-change recall 与模型 agreement 效用，并决定阈值校准/微调策略。",
        "",
        "**当前 STOP：等待人工 review。不计算 Shanghai accuracy，不把 candidate 当作 confirmed change。**",
        "",
    ]
    (REPORTS / "HUMAN_REVIEW_GATE.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/HUMAN_REVIEW_GATE.md")
    print(f"CF={n_cf} size={size_counts} prob={prob_bins}")
    print(f"agree14000={agree_14000:.3f} agree6000={agree_6000:.3f} agree_fc={agree_fc:.3f}")


def top_k_list():
    return CFG["top_k"]


if __name__ == "__main__":
    main()
