# Pre-GT Failure Analysis: Old Differencing vs Direct CD

- 生成时间：2026-08-18/19（Phase K）。比较对象：Old = SegFormer iter-6000 差分（A raw / C0.7 对象过滤 / C0.7+配准校正），Direct = changeformer、fc_siam_diff（WHU 训练 zero-shot）。
- **不计算 Shanghai 指标**（GT Gate 前禁止）；以下全部为结构/定性统计。

## 结构统计

| Method | Change area (ha) | Components | <30m² | 30-100m² | 100-500m² | 500m²-0.5ha | >=0.5ha | boundary share |
|---|---|---|---|---|---|---|---|---|
| A_grid | 275.75 | 53,137 | 41,570 | 5,885 | 4,650 | 1,012 | 20 | 36.4% |
| C0.7_grid | 87.35 | 5,853 | 0 | 3,819 | 1,739 | 290 | 5 | 10.5% |
| C0.7_shiftcorr | 93.31 | 6,531 | 39 | 4,326 | 1,863 | 298 | 5 | 13.7% |
| changeformer | 68.37 | 6,200 | 4,867 | 604 | 481 | 230 | 18 | 2.9% |
| fc_siam_diff | 1544.75 | 29,643 | 21,719 | 2,701 | 3,012 | 1,634 | 577 | 1.3% |

## 与 C0.7 旧差分的重叠（direct 方法）

| Direct method | IoU with C0.7 | old covered by direct | direct covered by old |
|---|---|---|---|
| changeformer | 0.0196 | 3.4% | 4.4% |
| fc_siam_diff | 0.0068 | 12.6% | 0.7% |

## 观察（候选结论，待 GT Gate 后验证）

- 直接 CD 的碎片化程度通常低于旧差分（见 components 与 <30m² 占比）。
- boundary share 反映变化是否集中在建筑边缘（配准伪变化特征）；直接 CD 若显著更低，说明其边缘伪变化抑制更好。
- 重叠低且方向一致的区域是可能真实变化；`direct_only` / `old_only` gallery 见 `outputs/failure_analysis/`。
- **ChangeFormer 零样本** change 面积（68.4 ha，占内容 0.89%，对比脚本按 Jan∩Apr 内容区限制；原始输出 1.11%）与旧 C0.7（87.4 ha，1.14%）同量级，但重叠仍很低（IoU≈0.02）：两种方法标记的变化区域基本不同——旧差分偏向边界/碎片，直接 CD 标记更整体的区域。该差异是否为真变化仍需 clean GT 判定。

## 重要 caveat

- OpenCD 直接 CD 输出是**二值 change map（无新增/拆除方向）**，因此直接 CD 的 gain/loss 拆分不适用（上表直接 CD 的 loss=0 是编码假象，不是"无拆除"）。
- **FC-Siam zero-shot（threshold 0.5）预测 change 占内容区 ~20.2%，明显过预测**；ChangeFormer 仅 0.89–1.11%，两者行为差异巨大，说明零样本跨域（克赖斯特彻奇→上海、季节差异）的可靠性与阈值/架构强相关——重校准或本地微调**必须依赖 clean Jan-Apr change GT（GT Gate）**。
- 直接 CD 的 boundary share 显著更低（1.3–2.9% vs 旧 C0.7 10.5%），说明其边缘伪变化更少；但 FC-Siam 存在 broad-area 过预测。
