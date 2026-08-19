"""Parse OpenCD training log -> WHU-CD results report (per-epoch val table + figure).

Usage:
  python scripts/whu_results_from_log.py --log <path.log> --tag fc_siam_diff
      [--test-json <optional test metrics json>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def parse_log(path: Path) -> list[dict]:
    rows = []
    cur = None
    pending = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Epoch\(val\)\s+\[(\d+)\]\[(\d+)/\d+\].*?aAcc:\s*([\d.]+)\s+mFscore:\s*([\d.]+)\s+mPrecision:\s*([\d.]+)\s+mRecall:\s*([\d.]+)\s+mIoU:\s*([\d.]+)", line)
            if m:
                cur = {
                    "epoch": int(m.group(1)),
                    "aAcc": float(m.group(3)),
                    "mFscore": float(m.group(4)),
                    "mPrecision": float(m.group(5)),
                    "mRecall": float(m.group(6)),
                    "mIoU": float(m.group(7)),
                }
                if pending is not None:
                    cur.update(pending)
                    pending = None
                rows.append(cur)
                continue
            if "|  changed" in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 6:
                    pending = {
                        "changed_Fscore": float(parts[1]),
                        "changed_Precision": float(parts[2]),
                        "changed_Recall": float(parts[3]),
                        "changed_IoU": float(parts[4]),
                    }
            elif "| unchanged" in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 6:
                    if pending is not None:
                        pending["unchanged_Fscore"] = float(parts[1])
                        pending["unchanged_IoU"] = float(parts[4])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--test-json", default=None)
    args = ap.parse_args()

    rows = parse_log(Path(args.log))
    if not rows:
        print("No val rows parsed. Check log format.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ep, [r["mIoU"] for r in rows], marker="o", label="mIoU")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("mIoU"); axes[0].set_title(f"{args.tag} val mIoU")
    axes[1].plot(ep, [r.get("changed_IoU", 0) for r in rows], marker="s", color="tab:red", label="changed IoU")
    axes[1].plot(ep, [r.get("changed_Fscore", 0) for r in rows], marker="^", color="tab:green", label="changed F1")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("metric"); axes[1].set_title("changed class")
    axes[1].legend()
    fig.tight_layout()
    fig_path = FIG_DIR / f"{args.tag}_whu_val_curves.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    best = max(rows, key=lambda r: r.get("changed_IoU", -1))
    md = [
        f"# WHU-CD {args.tag} Results",
        "",
        f"- 数据：WHU-CD-256（train 4,059 / val 779 / test 2,596，spatial split 见 `reports/whu_spatial_split.md`）。",
        f"- 日志：`{Path(args.log)}`",
        "",
        "## Validation 曲线（per epoch）",
        "",
        "| Epoch | mIoU | aAcc | mFscore | changed IoU | changed F1 | changed P | changed R |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['epoch']} | {r['mIoU']:.2f} | {r['aAcc']:.2f} | {r['mFscore']:.2f} | "
            f"{r.get('changed_IoU', float('nan')):.2f} | {r.get('changed_Fscore', float('nan')):.2f} | "
            f"{r.get('changed_Precision', float('nan')):.2f} | {r.get('changed_Recall', float('nan')):.2f} |"
        )
    md += [
        "",
        f"**Best changed IoU**: epoch {best['epoch']}, changed IoU {best.get('changed_IoU', 0):.2f}, "
        f"F1 {best.get('changed_Fscore', 0):.2f}, mIoU {best['mIoU']:.2f}.",
        "",
        "## All-unchanged collapse 排查记录",
        "",
    ]
    collapse = [r for r in rows if r.get("changed_IoU", 0) == 0]
    if collapse:
        md.append(
            "- 观察到了 all-unchanged 早期现象（epoch "
            + ", ".join(str(r["epoch"]) for r in collapse)
            + " val changed IoU=0）；随后恢复正常（见上表与 `reports/progress.md`）。"
        )
    else:
        md.append(
            f"- **未观察到 all-unchanged collapse**：从首个验证点（epoch {rows[0]['epoch']}）起 "
            "changed IoU 即为正（>0），模型正常学到了 changed 类。"
        )
    md += [
        "",
        f"Val curves: `outputs/figures/{args.tag}_whu_val_curves.png`",
        "",
    ]
    if args.test_json:
        t = json.loads(Path(args.test_json).read_text(encoding="utf-8"))
        md.append("## Test metrics（官方 test split）")
        md.append("")
        md.append("```json")
        md.append(json.dumps(t, ensure_ascii=False, indent=2))
        md.append("```")
        md.append("")
    if args.tag == "fc_siam_diff":
        out_name = "whu_fcsn_results.md"
    elif args.tag == "changeformer":
        out_name = "whu_changeformer_results.md"
    else:
        out_name = f"whu_{args.tag}_results.md"
    (REPORTS / out_name).write_text("\n".join(md), encoding="utf-8")
    print("wrote", REPORTS / out_name)


if __name__ == "__main__":
    main()
