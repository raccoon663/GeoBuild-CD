# Model Provenance Audit

> Port of the project's provenance audit (originally 2026-08-18). Local
> filesystem paths have been replaced with generic placeholders
> (`<LOCAL_SHANGHAI_DATA>`, `<MODEL_ROOT>`, `<LEGACY_PROJECT>`, ...). See
> [`data_and_splits.md`](data_and_splits.md) for the placeholder legend.

## Summary

- **Main local supervision:** April imagery and labels.
- **January labels:** independent temporal evaluation data (not a training
  reference month).
- **Local checkpoint:** iter-6000 (B5 shadow finetune,
  `segformer_b5_zhelin_local_best_20260624.pth`).
- **Original packaged app:** WHU-generalized iter-14000.
- **ONNX lineage verified by weight/output comparison:** all ONNX files
  shipped with the legacy app are byte-identical to the iter-14000 weights —
  **not** the iter-6000 local model used by the GeoBuild-CD pure-PyTorch
  inference.

---

## Detailed audit notes

*Historical working notes are retained verbatim (Chinese) so the evidence
chain stays auditable; numbers and evidence are unchanged.*

### 1. Key conclusions

1. 最终本地 checkpoint `segformer_b5_zhelin_local_best_20260624.pth`（即 `segformer_b5_shadow_finetune_20260624/best_mIoU_iter_6000.pth`）训练自 **April 影像 + April 建筑标签**（经 `<LOCAL_LABEL_RASTER>` ENVI 标签栅格切瓦），叠加 hard-negative 与 shadow 样本。
2. 训练链：SegFormer-B2 (April 原始集) → B2 hard finetune → B5 hard finetune → **B5 shadow finetune（iter 6000 = 最终本地模型）** → B5 WHU-generalized finetune（iter 14000）。
3. **关键修正**：仓库内三份 ONNX（旧 App 的 `model.onnx`、`segformer_b5_shadow_finetune_20260624.onnx`、`segformer_b5_whu_generalized_best_mIoU_iter_14000.onnx`）字节完全相同，逐像素验证表明它们对应 **WHU-generalized iter-14000 权重**，与 GeoBuild-CD 纯 PyTorch 推理所用的 iter-6000 本地模型**不是同一模型**。此前 "torch vs ONNX ~0.7% 差异" 实为两个不同 checkpoint 的真实差异，而非纯数值误差。

### 2. Supervision data source (April)

**April imagery**

- 路径：`<LOCAL_SHANGHAI_DATA>/2026-04.tif`
- 尺寸：31153 × 21385，4 波段 uint8（RGBN），EPSG:4326
- 像素：4.492e-6°（约 0.5 m），origin (121.37455833, 30.89216834)
- nodata 陷阱：边框像素值 1（内容掩膜处理见 [`docs/methodology.md`](methodology.md)）

**April labels**

人工标注工程（ArcGIS Pro）：`<LOCAL_ARCGIS_PROJECT>`，要素类 `landuse`（别名"建筑"），2026-05-08 创建，2026-06-16 仍有编辑，字段 `Value=1` 表示建筑。

| SHP | 导出日期 | 记录数 |
|---|---|---:|
| `<LOCAL_SHANGHAI_DATA>/Building_labeling.shp` | 2026-05-25 | 5397 |
| `<LOCAL_SHANGHAI_DATA>/2026-04_labels.shp` | 2026-05-13 导出 / 2026-06-16 更新 | 5386 |
| `<LOCAL_SHANGHAI_DATA>/buildingfinal.shp` | 2026-05-25 | 5429 |

**ENVI label raster (strictly aligned with April imagery)**

- 路径：`<LOCAL_LABEL_RASTER>`（`labeledRaster.hdr`）
- 尺寸/transform 与 April 影像完全一致（rasterio 实测同 grid），5 波段，**band 5 = Label Mask**，building 值 = 1，ignore = 255。
- 验证：抽查 6 个含建筑瓦片，`data/buildings/ann_dir/train/*.png` 与 band 5 的逐像素 IoU = 1.000 → 训练监督即该 April 标签栅格。

### 3. Training data composition and tiling evidence

**`data/buildings` (original April training set)**

- 生成脚本：`tile_buildings_dataset.py`（April 栅格 RGB + labeledRaster band5，512 步长切瓦，按列 70/15/15 分 train/val/test）
- 瓦片计数：train 907 / val 223 / test 172
- **月份证据**：瓦片命名 `tile_r{row}_c{col}.png`，train 中 max row = 20480、max col = 30208，恰好是 31153 × 21385（April）的 512 网格；March (33500×23955) 应为 23443/32988，January (30024×21220) 应为 20708/29512，均不吻合。
- 目录：`<LEGACY_PROJECT>/data/buildings`

**`data/buildings_clean_eval` (clean eval set, April)**

- 10 个 ~2000×2000 区域（4 val + 6 test），由 `build_clean_eval_from_shapes.py` 从 April 栅格 + 建筑 SHP 构建；region manifest bounds 全部落在 April 范围 (121.3746–121.5145, 30.7961–30.8922) 内。
- val 36 tiles / test 54 tiles。旧项目报告 Building IoU 0.749 的 54-patch 测试集即此 April clean-eval test 集。

**`data/buildings_hard_train` (hard negatives)**

- 2 个区域 × 9 tiles = 18 tiles（`region_manifest.csv`），bounds 位于镇区，来自 hard-area 样本（`<LOCAL_SHANGHAI_DATA>/HardArea.shp`、`<LOCAL_SHANGHAI_DATA>/clipshp.shp` 等为派生/剪裁产物）。

**`shadow_b5_labelled_20260624` (shadow samples)**

- 4 个区域 × 9 tiles = 36 tiles（`<LEGACY_PROJECT>/shadow_b5_labelled_20260624/region_manifest.csv`），window 坐标均在 April 栅格范围内；shadow finetune 中重复 10 倍。

**WHU building-segmentation data (generalized finetune only)**

- `<WHU_SEG_ROOT>`（train/val/test image+label）用于 2026-07-08 的 WHU-generalized finetune，不是本地 April 监督。

### 4. Training chain and checkpoint lineage

| Stage | 配置 | 初始化 | 训练数据 | 计划 iters | 保存 best | val mIoU* |
|---|---|---|---|---|---:|---:|
| B2 buildings | `segformer_mit-b2_buildings_512.py` | ADE20K B2 预训练 | `data/buildings` | 80000 | iter 24000 | ~82.4 (buildings val) |
| B2 hard finetune | `segformer_mit-b2_hard_finetune.py` | B2 best 24000 | buildings + hard | 8000 | iter 8000 | ~88.9 (clean_eval val) |
| B5 hard finetune | `segformer_mit-b5_hard_finetune.py` | ADE20K B5 预训练 | buildings + hard | 12000 | iter 12000 | ~86.3 (clean_eval val) |
| B5 shadow finetune | `segformer_mit-b5_shadow_finetune_20260624.py` | B5 hard best 12000 | buildings + hard + shadow×10 | 6000 | **iter 6000** | ~87.3 (clean_eval val) |
| B5 WHU-generalized | `segformer_mit-b5_whu_generalized_finetune.py` | zhelin local (iter 6000) | WHU + buildings + hard + shadow | 20000 | iter 14000 | ~92.0 (WHU+local val) |

*val mIoU 取自各次训练 `vis_data/*.json` 的 val 记录；不同 stage 的 val 集不同（B2 用 buildings val，后续用 clean_eval / WHU+local），不能直接横向比较。

**Final local model:**

- `<MODEL_ROOT>/segformer_b5_zhelin_local_best_20260624.pth` ≡ `<LEGACY_OUTPUTS>/segformer_b5_shadow_finetune_20260624/best_mIoU_iter_6000.pth`（state-dict SHA256 前缀一致，逐层权重一致）
- GeoBuild-CD 本地副本：`checkpoints/segformer_b5_zhelin_local_best_20260624.pth`

**WHU-generalized model:**

- `<LEGACY_OUTPUTS>/segformer_b5_whu_generalized_finetune/best_mIoU_iter_14000.pth`（与 iter-6000 权重不同，SHA256 前缀不同）

### 5. ONNX lineage verification (important correction)

以下 4 个 ONNX 文件 **MD5 完全相同（771DA32E10EBDE60E7A56B1C20CD5E85）**：

1. `third_party/legacy_change_detector_app/assets/model/model.onnx`（旧 App 打包）
2. `<MODEL_ROOT>/segformer_b5_whu_generalized_best_mIoU_iter_14000.onnx`
3. `checkpoints/segformer_b5_shadow_finetune_20260624.onnx`
4. `checkpoints/segformer_b5_whu_generalized_best_mIoU_iter_14000.onnx`

逐像素验证（5 个 April 512 瓦片，ONNX Runtime CPU vs 纯 PyTorch）：

| 对比 | mask 一致率 | mean \|Δp\| |
|---|---:|---:|
| ONNX vs iter-14000 | 100.00% | 0.00000 |
| ONNX vs iter-6000 | 95.79–100%（含建筑瓦片 95.79–99.26%） | 0.006–0.037 |

结论：

- 所有 ONNX 实际对应 **WHU-generalized iter-14000** 权重（字节级同权，可精确复现输出）。
- 文件名 `segformer_b5_shadow_finetune_20260624.onnx` 具有误导性：内容并非 iter-6000 本地模型。
- 历史 torch-vs-ONNX 对比脚本报告的 ~0.7% mask flip 是 **iter-6000 vs iter-14000 两个模型的真实差异**，不是数值误差；其 "数值差异" 解释需要纠正。
- `model.onnx.bak_20260709_0920`（MD5 726FD05B...）为 2026-07-09 替换前的旧 ONNX，未单独验证（推测为更早的 iter-6000 导出）。

**Implications**

- 旧 App（打包版）的变化检测实际使用 **WHU-generalized iter-14000** 模型。
- GeoBuild-CD 各月推理 `outputs/segmentation/*`（`scripts/predict_month.py`）使用 **iter-6000 本地模型**。
- 后续 "Old SegFormer difference vs Direct CD" 对比必须标注所用权重；若要忠实复现旧 App，应使用 iter-14000 torch 权重（可精确复现 ONNX 输出）。

### 6. Statements that were corrected

- 本项目 `README.md`、`configs/project.yaml`：reference month 由 2026-01 改为 **2026-04（April，训练监督月）**；January 描述为 **independent temporal evaluation data**。
- 旧的 "reference-month SegFormer reproduction" 实为 "April 监督模型在 January 独立标签上的跨期评估"，不是训练参考月复现。

### 7. Evidence file inventory

- 训练配置：`<LEGACY_OUTPUTS>/segformer_b5_shadow_finetune_20260624/segformer_mit-b5_shadow_finetune_20260624.py`；`<LEGACY_OUTPUTS>/segformer_b5_whu_generalized_finetune/segformer_mit-b5_whu_generalized_finetune.py`；`<LEGACY_PROJECT>/configs/buildings/*.py`
- 训练日志：`.../20260624_112611/vis_data/20260624_112611.json`；`.../20260708_142343/vis_data/20260708_142343.json`；原工程 `work/experiments/segformer_*/**/vis_data/*.json`
- 数据清单：`data/buildings*`、`<LEGACY_PROJECT>/shadow_b5_labelled_20260624/region_manifest.csv`
- 标签栅格：`<LOCAL_LABEL_RASTER>.hdr`
- ArcGIS 标注工程元数据：`<LOCAL_SHANGHAI_DATA>/2026-04_labels.shp.xml`、`<LOCAL_SHANGHAI_DATA>/Building_labeling.shp.xml`
- 哈希/验证：历史 torch-vs-ONNX 对比脚本（本地审计工具，未随仓库分发；其结论需按第 5 节修正）；本次核查脚本临时存放于本机 temp。
