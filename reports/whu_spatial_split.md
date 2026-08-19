# WHU-CD Spatial Split (OpenCD WHU-CD-256 build)

- 构建脚本：`scripts/whu_cd_prepare.py`；输出目录：`<WHU_CD_ROOT>/`（A/B/label + list/*.txt，由脚本生成）。
- 规则：官方整景（train mosaic 21243×15354，test mosaic 11265×15354）按全局 256 px 网格切瓦；train mosaic 再按 4×3 空间网格分区，**val = 块 (0,3) 与 (2,0)**，其余为 train；test mosaic 全部为 test。**无随机 patch split**。
- A=2012、B=2016、label=官方 change_label（0/255 PNG）；瓦片名 = `{split}_{row}_{col}.png`。

## 瓦片统计

| Split | Tiles | Change fraction (pixel) |
|---|---|---|
| train | 4059 | 4.3430% |
| val | 779 | 5.4270% |
| test | 2596 | 3.6830% |

## Train mosaic 4×3 块明细

| Block (row,col) | Split | Tiles | Change fraction |
|---|---|---|---|
| (0,0) | train | 420 | 0.5839% |
| (0,1) | train | 420 | 3.9342% |
| (0,2) | train | 420 | 3.4264% |
| (0,3) | val | 380 | 6.5357% |
| (1,0) | train | 420 | 3.9726% |
| (1,1) | train | 420 | 2.3930% |
| (1,2) | train | 420 | 4.5227% |
| (1,3) | train | 380 | 12.6452% |
| (2,0) | val | 399 | 4.3719% |
| (2,1) | train | 399 | 1.5246% |
| (2,2) | train | 399 | 1.0815% |
| (2,3) | train | 361 | 10.7271% |

## 说明

- 保留全部瓦片（含无变化瓦片），以保持官方类别分布；changed 类不平衡在训练阶段用 loss weighting / oversampling 处理（Phase G/H）。
- change_label 一致性见 `reports/whu_data_audit.md`（XOR vs 官方 change_label 一致率 ~98%）；**以官方 change_label 为准**。
- 空间划分图：`figures/whu_spatial_split.png`。
