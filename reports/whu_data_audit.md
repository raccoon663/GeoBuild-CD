# WHU-CD Data Audit



- 来源：WHU 官方 `Building change detection dataset_add.zip`（5.84 GB，已下载并解压到 `<WHU_CD_ROOT>/`）。

- 本审计只读不改；**不先切瓦、不清理标签**；空间划分见 `reports/whu_spatial_split.md`。



## 整景影像（mosaics）



| Mosaic | W x H | Bands | CRS | Pixel (m) | Origin (x, y) | Building frac (sampled) |

|---|---|---|---|---|---|---|

| 2012_train | 21243 x 15354 | 3 | EPSG:2193 | [0.2, 0.2] | [1560160.0, 5179409.2] | 0.1511 |

| 2012_test | 11265 x 15354 | 3 | EPSG:2193 | [0.2, 0.2] | [1564408.4, 5179409.2] | 0.1517 |

| 2016_train | 21243 x 15354 | 3 | EPSG:2193 | [0.2, 0.2] | [1560160.0, 5179409.25] | 0.1886 |

| 2016_test | 11265 x 15354 | 3 | EPSG:2193 | [0.2, 0.2] | [1564408.4, 5179409.25] | 0.1831 |



## 预切 512 瓦片（splited_images，2012/2016 同名配对）



| Split | 2012 images | 2016 images | 2012 labels | 2016 labels | name sets equal |

|---|---|---|---|---|---|

| train | 1260 | 1260 | 1260 | 1260 | True |

| test | 690 | 690 | 690 | 690 | True |



## Shapefiles（场景范围）



| Split | Records | Shape type | BBox | Fields |

|---|---|---|---|---|

| train | 1 | POLYGON | [1560159.9, 5176338.5, 1564408.5, 5179409.3] | Id |

| test | 1 | POLYGON | [1564408.5, 5176338.5, 1566661.3, 5179409.3] | Id |



## Change label 一致性（decimated 采样）



| Split | XOR vs change_label 一致率 | change frac | new-only frac | demo-only frac | change∩new | change∩demo |

|---|---|---|---|---|---|---|

| train | 97.7850% | 4.5860% | 5.0430% | 1.3050% | 87.2210% | 7.8380% |

| test | 98.1100% | 3.6880% | 4.2000% | 1.0440% | 89.8800% | 5.5820% |



## 结论 / 风险



- 2012/2016 同名整景与瓦片一一对应，transform 一致 → **A/B 配对对齐成立**（同一 grid）。

- 若 XOR 与官方 change_label 一致，则在构建 OpenCD 数据时可直接用 XOR(2012,2016) 作为 change label（或直接裁剪官方 change_label）；若有差异，必须记录并按官方 change_label 为准。

- change_label 为 uint8 {0,1}（1=change），0.2 m 像素；OpenCD WHU_CD_Dataset 的 `format_seg_map='to_binary'` 会把 <128 归 0、>=128 归 1。

- 注意：change 若主要等于 new-only（2016 新增），说明该数据集变化定义偏向“新增建筑”；拆除部分占比见上表。

