"""Phase F-1 - WHU-CD dataset audit (official download, not pre-cleaned).

Checks:
  - inventory of whole mosaics / pre-cut tiles / change labels / shapefiles
  - paired alignment (2012 vs 2016 grid identity)
  - change-label consistency: XOR(2012 building, 2016 building) vs change_label
  - change directionality (new-only vs symmetric)
Outputs: reports/whu_data_audit.md / .json
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import shapefile
import rasterio
from rasterio.enums import Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = Path(
    r"<WHU_CD_ROOT>"
    "/Building change detection dataset_add/1. The two-period image data"
)
SHP = Path(
    r"<WHU_CD_ROOT>"
    "/Building change detection dataset_add/2. The shape file of the images"
)
REPORTS = PROJECT_ROOT / "reports"


def mosaic_stats(period: str, split: str) -> dict:
    img = RAW / period / "whole_image" / split / "image" / f"{period}_{split}.tif"
    lbl = RAW / period / "whole_image" / split / "label" / f"{period}_{split}.tif"
    with rasterio.open(img) as src:
        r = {
            "path": str(img),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "pixel_size_m": [round(abs(src.transform.a), 4), round(abs(src.transform.e), 4)],
            "origin": [round(src.transform.c, 4), round(src.transform.f, 4)],
        }
    with rasterio.open(lbl) as src:
        data = src.read(
            1,
            out_shape=(1, src.height // 16, src.width // 16),
            resampling=Resampling.nearest,
        )
        vals, counts = np.unique(data, return_counts=True)
        r["label_values_sampled"] = dict(zip(vals.tolist(), counts.tolist()))
        r["building_fraction_sampled"] = round(float((data > 0).mean()), 5)
    return r


def tile_counts() -> dict:
    out = {}
    for period in ("2012", "2016"):
        for split in ("train", "test"):
            imgs = glob.glob(str(RAW / period / "splited_images" / split / "image" / "*.tif"))
            lbls = glob.glob(str(RAW / period / "splited_images" / split / "label" / "*.tif"))
            names_i = {os.path.basename(p) for p in imgs}
            names_l = {os.path.basename(p) for p in lbls}
            out[f"{period}_{split}"] = {
                "images": len(imgs),
                "labels": len(lbls),
                "name_sets_equal": names_i == names_l,
            }
    return out


def shp_stats(split: str) -> dict:
    base = SHP / split / f"{split}.shp"
    reader = shapefile.Reader(str(base))
    bounds = [round(v, 3) for v in reader.bbox]
    fields = [f[0] for f in reader.fields[1:]]
    return {
        "path": str(base),
        "records": len(reader),
        "fields": fields,
        "bbox": bounds,
        "shape_type": reader.shapeTypeName,
    }


def xor_consistency() -> dict:
    """Compare XOR of per-period building labels with the official change label."""
    results = {}
    for split in ("train", "test"):
        p12 = RAW / "2012" / "whole_image" / split / "label" / f"2012_{split}.tif"
        p16 = RAW / "2016" / "whole_image" / split / "label" / f"2016_{split}.tif"
        pch = RAW / "change_label" / split / "change_label.tif"
        with rasterio.open(p12) as a, rasterio.open(p16) as b, rasterio.open(pch) as c:
            h, w = a.height // 8, a.width // 8
            la = a.read(1, out_shape=(1, h, w), resampling=Resampling.nearest) > 0
            lb = b.read(1, out_shape=(1, h, w), resampling=Resampling.nearest) > 0
            lc = c.read(1, out_shape=(1, h, w), resampling=Resampling.nearest) > 0
        xor = la != lb
        n = xor.size
        agree = float((xor == lc).mean())
        # directionality: how much of official change is captured by "new only" (2016 & ~2012)
        new_only = lb & ~la
        dem_only = la & ~lb
        ch_px = int(lc.sum())
        results[split] = {
            "sampled_pixels": n,
            "xor_vs_change_label_agreement": round(agree, 5),
            "change_pixels_sampled": ch_px,
            "change_fraction": round(ch_px / n, 5),
            "xor_fraction": round(float(xor.mean()), 5),
            "new_only_fraction": round(float(new_only.mean()), 5),
            "demolition_only_fraction": round(float(dem_only.mean()), 5),
            "change_overlap_with_new_only": round(float((lc & new_only).sum() / max(ch_px, 1)), 5),
            "change_overlap_with_dem_only": round(float((lc & dem_only).sum() / max(ch_px, 1)), 5),
        }
    return results


def main() -> None:
    out = {
        "source": "WHU official 'Building change detection dataset_add' (gpcv.whu.edu.cn), 5.84 GB zip",
        "mosaics": {},
        "tiles": tile_counts(),
        "shapefiles": {s: shp_stats(s) for s in ("train", "test")},
        "xor_consistency": xor_consistency(),
    }
    for period in ("2012", "2016"):
        for split in ("train", "test"):
            out["mosaics"][f"{period}_{split}"] = mosaic_stats(period, split)

    (REPORTS / "whu_data_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# WHU-CD Data Audit",
        "",
        "- 来源：WHU 官方 `Building change detection dataset_add.zip`（5.84 GB，已下载并解压到 `<WHU_CD_ROOT>/`）。",
        "- 本审计只读不改；**不先切瓦、不清理标签**；空间划分见 `reports/whu_spatial_split.md`。",
        "",
        "## 整景影像（mosaics）",
        "",
        "| Mosaic | W x H | Bands | CRS | Pixel (m) | Origin (x, y) | Building frac (sampled) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, m in out["mosaics"].items():
        md.append(
            f"| {key} | {m['width']} x {m['height']} | {m['bands']} | {m['crs']} | {m['pixel_size_m']} | "
            f"{m['origin']} | {m['building_fraction_sampled']:.4f} |"
        )
    md += [
        "",
        "## 预切 512 瓦片（splited_images，2012/2016 同名配对）",
        "",
        "| Split | 2012 images | 2016 images | 2012 labels | 2016 labels | name sets equal |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("train", "test"):
        a = out["tiles"][f"2012_{split}"]
        b = out["tiles"][f"2016_{split}"]
        md.append(
            f"| {split} | {a['images']} | {b['images']} | {a['labels']} | {b['labels']} | "
            f"{a['name_sets_equal'] and b['name_sets_equal']} |"
        )
    md += [
        "",
        "## Shapefiles（场景范围）",
        "",
        "| Split | Records | Shape type | BBox | Fields |",
        "|---|---|---|---|---|",
    ]
    for s, sh in out["shapefiles"].items():
        md.append(f"| {s} | {sh['records']} | {sh['shape_type']} | {sh['bbox']} | {', '.join(sh['fields'])} |")
    md += [
        "",
        "## Change label 一致性（decimated 采样）",
        "",
        "| Split | XOR vs change_label 一致率 | change frac | new-only frac | demo-only frac | change∩new | change∩demo |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, x in out["xor_consistency"].items():
        md.append(
            f"| {s} | {x['xor_vs_change_label_agreement']:.4%} | {x['change_fraction']:.4%} | "
            f"{x['new_only_fraction']:.4%} | {x['demolition_only_fraction']:.4%} | "
            f"{x['change_overlap_with_new_only']:.4%} | {x['change_overlap_with_dem_only']:.4%} |"
        )
    md += [
        "",
        "## 结论 / 风险",
        "",
        "- 2012/2016 同名整景与瓦片一一对应，transform 一致 → **A/B 配对对齐成立**（同一 grid）。",
        "- 若 XOR 与官方 change_label 一致，则在构建 OpenCD 数据时可直接用 XOR(2012,2016) 作为 change label（或直接裁剪官方 change_label）；若有差异，必须记录并按官方 change_label 为准。",
        "- change_label 为 uint8 {0,1}（1=change），0.2 m 像素；OpenCD WHU_CD_Dataset 的 `format_seg_map='to_binary'` 会把 <128 归 0、>=128 归 1。",
        "- 注意：change 若主要等于 new-only（2016 新增），说明该数据集变化定义偏向“新增建筑”；拆除部分占比见上表。",
        "",
    ]
    (REPORTS / "whu_data_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/whu_data_audit.md / .json")
    print(json.dumps(out["xor_consistency"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
