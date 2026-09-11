┌──────────────────────────────────────────────┐
│                  输入阶段                    │
└──────────────────────────────────────────────┘

        Template ROI                  Data ROI
             │                            │
             │                            │
             └────────────┬───────────────┘
                          ▼
                输入两张 ROI 图像
                          │
                          ▼
              ┌─────────────────────┐
              │ 统一图像数据类型    │
              │     CV_8U            │
              └─────────────────────┘
                          │
                          ▼
                 生成 Gray 灰度图
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
          Template Gray        Data Gray
                 │                 │
                 ▼                 ▼
          Gray Histogram      Gray Histogram
                 │                 │
                 ▼                 ▼
        Histogram Threshold Histogram Threshold
                 │                 │
                 ▼                 ▼
          Template Binary       Data Binary
                 │                 │
                 └────────┬────────┘
                          ▼
                亚像素几何对位
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        Phase Correlation          ECC
              │                       │
              └───────────┬───────────┘
                          ▼
                  Sub-pixel Shift
                          │
                          ▼
                 Warp Data 原图
                          │
                          ▼
              计算有效共同 ROI
                          │
                          ▼
             Crop Common Valid ROI
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
       Template Valid ROI        Data Valid ROI
             │                         │
             └────────────┬────────────┘
                          ▼
                逐通道亮度归一化
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
       Template Normalize       Data Normalize
             │                         │
             └────────────┬────────────┘
                          ▼
                 Normalized Image
                          │
                          ▼
                Difference Calculation
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
        Absolute        Signed         L2
        Difference     Difference    Difference
             │            │            │
             │       ┌────┴────┐       │
             │       ▼         ▼       │
             │    diff > +T  diff < -T│
             │       │         │       │
             │       ▼         ▼       │
             │   Bright Mask Dark Mask │
             │       │         │       │
             │       └────┬────┘       │
             │            ▼            │
             └────────► Difference ◄───┘
                          │
                          ▼
                 Adaptive Threshold
                          │
                          ▼
             ┌────────────────────────┐
             │      Morphology        │
             │ Open / Close / Dilate  │
             │ Erode / Remove Noise   │
             └────────────────────────┘
                          │
                          ▼
               Connected Components
                          │
                          ▼
                    Defect Blob
                          │
                          ▼
               Feature Extraction
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
        Bright Defect             Dark Defect
             │                         │
             └────────────┬────────────┘
                          ▼
                  Defect Filtering
                          │
                          ▼
                 Final Defect List
				 
10 个核心阶段			 
01 ROI Input
02 CV_8U + Gray
03 Histogram + Binary
04 Sub-pixel Alignment
05 Warp + Common ROI
06 Per-channel Intensity Normalize
07 Difference
08 Bright / Dark Detection
09 Morphology + Blob
10 Defect Filtering