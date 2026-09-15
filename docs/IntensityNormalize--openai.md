                 IntensityNormalize
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   ① ROI统计        ② 灰度模型       ③ 参数计算
        │               │               │
        ↓               ↓               ↓
   Mean/Std/MinMax   Offset/Gain      Normalize
                                        │
                         ┌──────────────┴──────────────┐
                         ↓                             ↓
                    ④ Clipping                  ⑤ Mask
                         │                             │
                         └──────────────┬──────────────┘
                                        ↓
                                   ⑥ Output


IntensityNormalize 可以归纳为 8 大类：
| 类别                    | 核心思想           | 典型公式                | 工业 Die-to-Die |
| --------------------- | -------------- | ------------------- | ------------- |
| ① Offset              | 只调整亮度          | `I'=I+b`            | ★★★           |
| ② Gain                | 只调整对比度         | `I'=aI`             | ★★★           |
| ③ Gain + Offset       | 同时调整亮度/对比度     | `I'=aI+b`           | ★★★★★         |
| ④ Mean / Std          | 对齐均值和标准差       | `(I-μ)/σ`           | ★★★★★         |
| ⑤ Min-Max             | 对齐动态范围         | `(I-min)/(max-min)` | ★★★           |
| ⑥ Histogram           | 对齐灰度分布         | Histogram Mapping   | ★★★★          |
| ⑦ Percentile / Robust | 用分位数代替 Min/Max | P1/P99              | ★★★★          |
| ⑧ Local / Background  | 空间局部归一化        | `I-B(x,y)`          | ★★★★★         |


IntensityNormalize
│
├── Mean
│      → NPP
│
├── StdDev
│      → NPP
│
├── Min/Max
│      → NPP
│
├── Histogram
│      → NPP
│
├── Add / Sub
│      → NPP
│
├── Mul
│      → NPP
│
├── AbsDiff
│      → NPP
│
└── 特殊 Normalize
       → Custom CUDA Kernel



                IntensityNormalize
                        │
          ┌─────────────┴─────────────┐
          ↓                           ↓
      Statistics                 Spatial Model
          │                           │
    ┌─────┼─────┐                ┌────┴────┐
    ↓     ↓     ↓                ↓         ↓
   Mean  Std  Histogram       Global      Local
    │     │     │                │         │
    └─────┼─────┘                └────┬────┘
          ↓                           ↓
       Parameters                Background
          │                           │
          └────────────┬──────────────┘
                       ↓
                Pixel Transform
                       ↓
                    Clamp
                       ↓
                    Output


# 12. Strategy 6: Min-Max Normalize
# 18. Strategy 12: Background Subtraction
# 19. Strategy 13: Background Division
# 20. Strategy 14: Local Mean Normalize
# 21. Strategy 15: Local Mean / Std Normalize
