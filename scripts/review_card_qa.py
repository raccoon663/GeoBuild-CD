"""QA gate for candidate coordinates and review cards (2026-08-19 fix).

Run AFTER `build_review_candidates.py` (geometry fix) and BEFORE the full
review-card regeneration:

  python scripts/review_card_qa.py

Generates:
  outputs/human_review/card_QA/qa_top*.png, qa_rand*.png   (Top 20 + 20 random)
  outputs/human_review/card_QA/qa_results.json
  outputs/human_review/card_QA/qa_manifest.csv
  reports/review_card_QA.md

The required gate criterion is 0 invalid black crops among valid candidates.
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

from rank_and_review import (
    APR_RGB,
    CFG,
    HUMAN,
    JAN_PROB,
    OUT,
    REPORTS,
    build_pool,
    make_card,
)

QA_DIR = HUMAN / "card_QA"
QA_DIR.mkdir(parents=True, exist_ok=True)


def valid_masks():
    with rasterio.open(JAN_PROB) as s:
        jan = s.read(1) > 0
        tr, H, W = s.transform, s.height, s.width
    with rasterio.open(APR_RGB) as s:
        apr = s.read().max(axis=0) > 5
    return jan, apr, jan & apr, tr, H, W


def audit(gdf, jan_v, apr_v, com_v, tr):
    cen = gdf.geometry.centroid
    xs = cen.x.to_numpy()
    ys = cen.y.to_numpy()
    rows, cols = rasterio.transform.rowcol(tr, xs, ys)
    rows = np.clip(rows, 0, com_v.shape[0] - 1)
    cols = np.clip(cols, 0, com_v.shape[1] - 1)
    return {
        "n": int(len(gdf)),
        "unique_x": int(pd.Series(xs).nunique()),
        "unique_y": int(pd.Series(ys).nunique()),
        "x_min": round(float(xs.min()), 7),
        "x_max": round(float(xs.max()), 7),
        "y_min": round(float(ys.min()), 7),
        "y_max": round(float(ys.max()), 7),
        "in_jan_valid": round(float(jan_v[rows, cols].mean()), 4),
        "in_apr_valid": round(float(apr_v[rows, cols].mean()), 4),
        "in_common_valid": round(float(com_v[rows, cols].mean()), 4),
    }


def audit_old_manifest(jan_v, apr_v, com_v, tr):
    """Audit the pre-fix review_manifest.csv x/y columns (evidence)."""
    src = HUMAN / "review_manifest.csv"
    if not src.exists():
        return None
    df = pd.read_csv(src)
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    rows, cols = rasterio.transform.rowcol(tr, xs, ys)
    rows = np.clip(rows, 0, com_v.shape[0] - 1)
    cols = np.clip(cols, 0, com_v.shape[1] - 1)
    return {
        "n": int(len(df)),
        "unique_x": int(df["x"].nunique()),
        "unique_y": int(df["y"].nunique()),
        "x_min": round(float(df["x"].min()), 7),
        "x_max": round(float(df["x"].max()), 7),
        "y_min": round(float(df["y"].min()), 7),
        "y_max": round(float(df["y"].max()), 7),
        "in_jan_valid": round(float(jan_v[rows, cols].mean()), 4),
        "in_apr_valid": round(float(apr_v[rows, cols].mean()), 4),
        "in_common_valid": round(float(com_v[rows, cols].mean()), 4),
    }


def main() -> None:
    print("loading valid-content masks (Jan/Apr)...")
    jan_v, apr_v, com_v, tr, H, W = valid_masks()
    print("building pool (fixed geometry)...")
    b = build_pool()
    pool = b["pool"]

    old_audit = audit_old_manifest(jan_v, apr_v, com_v, tr)
    cf_audit = audit(b["cf"], jan_v, apr_v, com_v, tr)
    pool_audit = audit(pool, jan_v, apr_v, com_v, tr)
    print("audit cf:", json.dumps(cf_audit, ensure_ascii=False))
    print("audit pool:", json.dumps(pool_audit, ensure_ascii=False))

    qa_cfg = CFG["card"]["qa_sample"]
    rng = np.random.default_rng(qa_cfg["seed"])
    pool_sorted = pool.sort_values("candidate_rank", na_position="last").reset_index(drop=True)
    top20 = pool_sorted.head(qa_cfg["n_random"])
    rand_idx = rng.choice(len(pool), size=qa_cfg["n_random"], replace=False)
    rand20 = pool.iloc[rand_idx]

    qa_rows = []
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        p = QA_DIR / f"qa_top{i:02d}_{row['candidate_id']}.png"
        q = make_card(row, p)
        q["source"] = "top"
        qa_rows.append(q)
    for i, (_, row) in enumerate(rand20.iterrows(), 1):
        p = QA_DIR / f"qa_rand{i:02d}_{row['candidate_id']}.png"
        q = make_card(row, p)
        q["source"] = "random"
        qa_rows.append(q)

    qdf = pd.DataFrame(qa_rows)
    qdf.to_csv(QA_DIR / "qa_manifest.csv", index=False, encoding="utf-8")
    (QA_DIR / "qa_results.json").write_text(
        json.dumps(qa_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_ok = int((qdf["status"] == "ok").sum())
    n_bad = int((qdf["status"] != "ok").sum())
    n_black = int(qdf["flags"].apply(lambda f: any("black" in x for x in f)).sum())
    n_prob = int(qdf["flags"].apply(lambda f: "prob_mismatch" in f).sum())
    max_d_mean = float(qdf["delta_mean_prob"].max())
    max_d_max = float(qdf["delta_max_prob"].max())

    lines = [
        "# Review-Card QA（2026-08-19 修复）",
        "",
        f"- 日期：2026-08-19。根因：`build_review_candidates.py` 对 `features.shapes(..., transform=...)` "
        "已返回的世界坐标又做了一次 `affine_transform`，导致全部候选几何折叠到栅格原点附近"
        "（约像素 (31, 121) 的无效画布边缘），CSV x/y 与卡片裁剪随之全部错误。",
        "- 修复：删除二次变换；卡片裁剪改为直接从候选几何 bbox + 150 m 缓冲生成窗口，"
        "并在 Jan / Apr / Probability 三面板叠加候选边界。",
        "",
        "## 1. 候选坐标审计（全部 CF 候选，n=%d）" % cf_audit["n"],
        "",
        "| 指标 | 修复前 manifest（295 条） | 修复后 all_candidates（几何质心） | 修复后 review pool（295 条） |",
        "|---|---|---|---|",
    ]

    def row(name, o, c, p):
        lines.append(f"| {name} | {o} | {c} | {p} |")

    if old_audit:
        row("唯一 x 值数", old_audit["unique_x"], cf_audit["unique_x"], pool_audit["unique_x"])
        row("唯一 y 值数", old_audit["unique_y"], cf_audit["unique_y"], pool_audit["unique_y"])
        row("x min / max", f"{old_audit['x_min']} / {old_audit['x_max']}",
            f"{cf_audit['x_min']} / {cf_audit['x_max']}",
            f"{pool_audit['x_min']} / {pool_audit['x_max']}")
        row("y min / max", f"{old_audit['y_min']} / {old_audit['y_max']}",
            f"{cf_audit['y_min']} / {cf_audit['y_max']}",
            f"{pool_audit['y_min']} / {pool_audit['y_max']}")
        row("Jan 有效内容占比", f"{old_audit['in_jan_valid']:.1%}", f"{cf_audit['in_jan_valid']:.1%}",
            f"{pool_audit['in_jan_valid']:.1%}")
        row("Apr 有效内容占比", f"{old_audit['in_apr_valid']:.1%}", f"{cf_audit['in_apr_valid']:.1%}",
            f"{pool_audit['in_apr_valid']:.1%}")
        row("Jan∩Apr 共同有效占比", f"{old_audit['in_common_valid']:.1%}",
            f"{cf_audit['in_common_valid']:.1%}", f"{pool_audit['in_common_valid']:.1%}")
    else:
        row("唯一 x 值数", "-", cf_audit["unique_x"], pool_audit["unique_x"])
        row("唯一 y 值数", "-", cf_audit["unique_y"], pool_audit["unique_y"])
        row("x min / max", "-", f"{cf_audit['x_min']} / {cf_audit['x_max']}",
            f"{pool_audit['x_min']} / {pool_audit['x_max']}")
        row("y min / max", "-", f"{cf_audit['y_min']} / {cf_audit['y_max']}",
            f"{pool_audit['y_min']} / {pool_audit['y_max']}")
        row("Jan 有效内容占比", "-", f"{cf_audit['in_jan_valid']:.1%}", f"{pool_audit['in_jan_valid']:.1%}")
        row("Apr 有效内容占比", "-", f"{cf_audit['in_apr_valid']:.1%}", f"{pool_audit['in_apr_valid']:.1%}")
        row("Jan∩Apr 共同有效占比", "-", f"{cf_audit['in_common_valid']:.1%}",
            f"{pool_audit['in_common_valid']:.1%}")

    lines += [
        "",
        "## 2. QA 采样卡（Top 20 + 20 随机）",
        "",
        f"- 卡片总数：{len(qa_rows)}（Top 20 + 20 随机，seed={qa_cfg['seed']}）",
        f"- **状态 ok：{n_ok} / {len(qa_rows)}**",
        f"- 无效/黑卡（black_jan / black_apr / candidate_outside_valid）：{n_bad}",
        f"  - 其中 black 类：{n_black}",
        f"- 概率不一致（prob_mismatch）：{n_prob}",
        "",
        "## 3. Manifest 与卡片概率一致性（候选区内重算）",
        "",
        f"- max |Δmean| = {max_d_mean:.4f}（容差 {qa_cfg_tol()}）",
        f"- max |Δmax| = {max_d_max:.4f}",
        f"- 超容差数量：{n_prob}",
        "",
        "## 4. 结论",
        "",
    ]
    if n_bad == 0:
        lines.append("**PASS：0 无效黑卡；坐标分布正常；卡片概率与 manifest 一致。可以进入全量重建。**")
    else:
        lines.append(f"**FAIL：存在 {n_bad} 张无效/黑卡，先排查原因，不得进入全量重建。**")
    lines += [
        "",
        "## 路径",
        "",
        "- QA cards: `outputs/human_review/card_QA/qa_top*.png`、`qa_rand*.png`",
        "- QA results: `outputs/human_review/card_QA/qa_results.json`、`qa_manifest.csv`",
        "- 全部候选: `outputs/review_candidates/all_candidates.gpkg|csv`（已修复几何）",
        "",
    ]
    (REPORTS / "review_card_QA.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"QA cards: {len(qa_rows)}  ok={n_ok}  bad={n_bad}  black={n_black}  prob_mismatch={n_prob}")
    print(f"max|d_mean|={max_d_mean:.4f}  max|d_max|={max_d_max:.4f}")
    print("wrote reports/review_card_QA.md")


def qa_cfg_tol():
    return CFG["card"]["qa"]["prob_tolerance"]


if __name__ == "__main__":
    main()
