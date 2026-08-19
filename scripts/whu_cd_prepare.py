"""Phase F-2 - Build OpenCD-format WHU-CD-256 from official mosaics with a
deterministic spatial split (no random patch split).

Layout:
  - official train mosaic  -> 4x3 spatial grid, val = blocks {(0,3),(2,0)}
  - official test mosaic   -> all test
  - each block tiled at 256x256 non-overlap (global 256 grid)
  - A = 2012, B = 2016, label = official change_label (0/255 PNG)
  - outputs under third_party/open-cd/data/WHU-CD-256/{A,B,label}/ + list/*.txt
  - reports/whu_spatial_split.md + manifest json
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = Path(
    r"<WHU_CD_ROOT>"
    r"\Building change detection dataset_add\1. The two-period image data"
)
DST = PROJECT_ROOT / "third_party" / "open-cd" / "data" / "WHU-CD-256"
REPORTS = PROJECT_ROOT / "reports"
TILE = 256

# spatial grid over the train mosaic (px)
COL_EDGES = [0, 5376, 10752, 16128, 21243]
ROW_EDGES = [0, 5120, 10240, 15354]
VAL_BLOCKS = {(0, 3), (2, 0)}  # (row_block, col_block)


def block_of(col: int, row: int) -> tuple[int, int]:
    c = bisect.bisect_right(COL_EDGES, col) - 1
    r = bisect.bisect_right(ROW_EDGES, row) - 1
    return r, c


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (DST / "A").mkdir(parents=True, exist_ok=True)
    (DST / "B").mkdir(parents=True, exist_ok=True)
    (DST / "label").mkdir(parents=True, exist_ok=True)
    (DST / "list").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (DST / "list" / f"{split}.txt").write_text("", encoding="utf-8")

    paths = {
        "train": {
            "a": RAW / "2012" / "whole_image" / "train" / "image" / "2012_train.tif",
            "b": RAW / "2016" / "whole_image" / "train" / "image" / "2016_train.tif",
            "label": RAW / "change_label" / "train" / "change_label.tif",
        },
        "test": {
            "a": RAW / "2012" / "whole_image" / "test" / "image" / "2012_test.tif",
            "b": RAW / "2016" / "whole_image" / "test" / "image" / "2016_test.tif",
            "label": RAW / "change_label" / "test" / "change_label.tif",
        },
    }

    counts = {"train": 0, "val": 0, "test": 0}
    change_px = {"train": 0, "val": 0, "test": 0}
    total_px = {"train": 0, "val": 0, "test": 0}
    block_stats = {}

    for split, paths_ in paths.items():
        with rasterio.open(paths_["a"]) as a, rasterio.open(paths_["b"]) as b, rasterio.open(paths_["label"]) as lab:
            h, w = a.height, a.width
            assert b.width == w and b.height == h
            assert lab.width == w and lab.height == h
            cols = list(range(0, w - TILE + 1, TILE))
            rows = list(range(0, h - TILE + 1, TILE))
            for row in rows:
                for col in cols:
                    if split == "test":
                        split_name = "test"
                    else:
                        rb, cb = block_of(col, row)
                        split_name = "val" if (rb, cb) in VAL_BLOCKS else "train"
                    name = f"{split_name}_{row:05d}_{col:05d}.png"
                    win = Window(col, row, TILE, TILE)
                    img_a = a.read([1, 2, 3], window=win)  # (3,256,256)
                    img_b = b.read([1, 2, 3], window=win)
                    lab_d = lab.read(1, window=win)
                    Image.fromarray(np.moveaxis(img_a, 0, -1)).save(DST / "A" / name)
                    Image.fromarray(np.moveaxis(img_b, 0, -1)).save(DST / "B" / name)
                    lab_png = np.where(lab_d > 0, 255, 0).astype(np.uint8)
                    Image.fromarray(lab_png).save(DST / "label" / name)
                    with (DST / "list" / f"{split_name}.txt").open("a", encoding="utf-8") as f:
                        f.write(name + "\n")
                    counts[split_name] += 1
                    total_px[split_name] += TILE * TILE
                    change_px[split_name] += int((lab_d > 0).sum())
                    if split == "train":
                        key = (rb, cb)
                        bl = block_stats.setdefault(
                            key, {"n_tiles": 0, "change_px": 0, "split": split_name}
                        )
                        bl["n_tiles"] += 1
                        bl["change_px"] += int((lab_d > 0).sum())

    # manifest
    manifest = {
        "tile_size": TILE,
        "counts": counts,
        "change_fraction": {
            k: round(change_px[k] / max(total_px[k], 1), 5) for k in counts
        },
        "train_blocks_4x3": {
            f"{r}_{c}": {**v, "change_fraction": round(v["change_px"] / max(v["n_tiles"] * TILE * TILE, 1), 5)}
            for (r, c), v in sorted(block_stats.items())
        },
        "val_blocks": sorted(VAL_BLOCKS),
        "grid_edges_px": {"cols": COL_EDGES, "rows": ROW_EDGES},
        "note": "Spatial split: official train mosaic partitioned into 4x3 blocks; val = blocks (0,3),(2,0). Test mosaic = test. No random patch split.",
    }
    (REPORTS / "whu_spatial_split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # figure: block layout
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = {"train": "#2f7f2f", "val": "#d62728", "test": "#1f77b4"}
    for (r, c), v in sorted(block_stats.items()):
        x0 = COL_EDGES[c] / 1000.0
        y0 = ROW_EDGES[r] / 1000.0
        w_ = (COL_EDGES[c + 1] - COL_EDGES[c]) / 1000.0
        h_ = (ROW_EDGES[r + 1] - ROW_EDGES[r]) / 1000.0
        ax.add_patch(plt.Rectangle((x0, y0), w_, h_, facecolor=cmap[v["split"]], alpha=0.7, edgecolor="k"))
        ax.text(x0 + w_ / 2, y0 + h_ / 2, f"({r},{c})\n{v['n_tiles']}", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, COL_EDGES[-1] / 1000)
    ax.set_ylim(0, ROW_EDGES[-1] / 1000)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("WHU-CD train mosaic spatial split (4x3 blocks; red=val)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(REPORTS.parent / "outputs" / "whu_spatial_split.png", dpi=120)
    plt.close(fig)

    md = [
        "# WHU-CD Spatial Split (OpenCD WHU-CD-256 build)",
        "",
        "- 构建脚本：`scripts/whu_cd_prepare.py`；输出目录：`third_party/open-cd/data/WHU-CD-256/`（A/B/label + list/*.txt）。",
        "- 规则：官方整景（train mosaic 21243×15354，test mosaic 11265×15354）按全局 256 px 网格切瓦；train mosaic 再按 4×3 空间网格分区，**val = 块 (0,3) 与 (2,0)**，其余为 train；test mosaic 全部为 test。**无随机 patch split**。",
        "- A=2012、B=2016、label=官方 change_label（0/255 PNG）；瓦片名 = `{split}_{row}_{col}.png`。",
        "",
        "## 瓦片统计",
        "",
        "| Split | Tiles | Change fraction (pixel) |",
        "|---|---|---|",
        f"| train | {counts['train']} | {manifest['change_fraction']['train']:.4%} |",
        f"| val | {counts['val']} | {manifest['change_fraction']['val']:.4%} |",
        f"| test | {counts['test']} | {manifest['change_fraction']['test']:.4%} |",
        "",
        "## Train mosaic 4×3 块明细",
        "",
        "| Block (row,col) | Split | Tiles | Change fraction |",
        "|---|---|---|---|",
    ]
    for (r, c), v in sorted(block_stats.items()):
        frac = v["change_px"] / max(v["n_tiles"] * TILE * TILE, 1)
        md.append(f"| ({r},{c}) | {v['split']} | {v['n_tiles']} | {frac:.4%} |")
    md += [
        "",
        "## 说明",
        "",
        "- 保留全部瓦片（含无变化瓦片），以保持官方类别分布；changed 类不平衡在训练阶段用 loss weighting / oversampling 处理（Phase G/H）。",
        "- change_label 一致性见 `reports/whu_data_audit.md`（XOR vs 官方 change_label 一致率 ~98%）；**以官方 change_label 为准**。",
        "- 空间划分图：`outputs/whu_spatial_split.png`。",
        "",
    ]
    (REPORTS / "whu_spatial_split.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("wrote reports/whu_spatial_split.md")


if __name__ == "__main__":
    main()
