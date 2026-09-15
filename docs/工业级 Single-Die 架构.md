                    Single Die
                        │
              ┌─────────┴─────────┐
              │                   │
        Global Statistics    Local Pattern
              │                   │
       Histogram Model       Cell/Pattern Model
              │                   │
              └─────────┬─────────┘
                        ↓
                 Reference Model
                        ↓
                 Registration
                        ↓
                 Normalization
                        ↓
              Difference / Residual
                        ↓
              Candidate Generation
                        ↓
              Morphology / Filtering
                        ↓
              Feature Extraction
                        ↓
              Defect Classification
                        ↓
                 Final Defect
				 
				 
                 Industrial Die Inspection
                           │
             ┌─────────────┴─────────────┐
             │                           │
           D2D                           SDI
             │                           │
       Die vs Die                  Cell vs Cell
             │                           │
       Reference Die               Same Die
             │                           │
             └─────────────┬─────────────┘
                           │
                    Registration
                           │
                    Difference Map
                           │
                  Normalization
                           │
                  Adaptive Threshold
                           │
                   Defect Candidate
                           │
                  Morphology / Blob
                           │
                 Defect Classification
				 
				 
				 | Wafer 类型                   | 首选检测算法                                              | 辅助算法                                          | 推荐程度  |
| -------------------------- | --------------------------------------------------- | --------------------------------------------- | ----- |
| **无图案 Wafer**              | **Surface / Global Background + Defect Detection**  | Dark/Bright Field、Adaptive Threshold、Blob/形态学 | ⭐⭐⭐⭐⭐ |
| **有图案 Wafer**              | **Pattern-Aware / Single-Die / Cell-to-Cell**       | D2D、局部背景建模、Edge/Contour                       | ⭐⭐⭐⭐⭐ |
| **有图案 + Golden Reference** | **Golden Die-to-Die / Golden Reference Comparison** | SSIM、像素差、梯度差、归一化差分                            | ⭐⭐⭐⭐⭐ |
| **有图案 + 同一 Wafer 多个 Die**  | **Die-to-Die + Multi-Die Consensus + Single-Die**   | Median Reference、Voting、Cluster               | ⭐⭐⭐⭐⭐ |
| **整片宏观缺陷**                 | **Wafer-Level Macro Inspection**                    | Background、低频、高亮/暗斑、边缘、Scratch                | ⭐⭐⭐⭐⭐ |
