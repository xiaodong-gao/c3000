# IntensityNormalize Algorithm Specification

> Version: 1.0  
> Module: `IntensityNormalize`  
> Target: Industrial Die-to-Die / Single Die Inspection  
> Language: C++ / CUDA C++  
> Data Type: `CV_8U`, `CV_16U`, `CV_32F`  
> Primary Target: High-throughput GPU inspection pipeline

---

# 1. Purpose

`IntensityNormalize` 用于消除 Template Die 与 Data Die 之间由于非缺陷因素造成的灰度差异，使两张图像在进入 Difference / Threshold / Defect Detection 之前具有更高的可比性。

主要解决：

- 光源亮度变化
- 相机曝光变化
- 相机 Gain 变化
- Die-to-Die 整体亮度变化
- 对比度变化
- 背景亮度变化
- 光照不均
- 局部渐晕
- ROI 灰度分布差异

核心目标：

```text
Template Image
      +
Data Image
      ↓
Intensity Normalization
      ↓
Reduced Non-defect Intensity Difference
      ↓
More Stable Die-to-Die Detection
```

---

# 2. Scope

本模块支持以下 Normalize 策略：

```text
1. Offset Normalize
2. Gain Normalize
3. Gain + Offset Normalize
4. Mean Normalize
5. Mean / StdDev Normalize
6. Min-Max Normalize
7. Percentile Normalize
8. Histogram Normalize
9. Histogram Matching
10. Robust Mean Normalize
11. Robust StdDev Normalize
12. Background Subtraction
13. Background Division
14. Local Mean Normalize
15. Local Mean / Std Normalize
16. Gamma Normalize
17. Log Normalize
18. Z-Score Normalize
19. Masked Normalize
20. Defect-Excluded Normalize
21. Iterative Normalize
```

实际生产版本必须通过配置选择策略，不允许多个策略无条件叠加。

---

# 3. Design Principle

## 3.1 Core Principle

IntensityNormalize 的本质是建立：

```text
I_normalized = F(I)
```

使：

```text
Distribution(Data_normalized)
≈
Distribution(Template)
```

但必须满足：

```text
真实缺陷不能被 Normalize 消除
```

因此：

> Normalize 的优先级是降低系统性灰度差异，而不是最大程度让两张图“看起来一样”。

---

# 4. Input / Output

## 4.1 Input

### Template

```cpp
const Image& templateImage;
```

### Data

```cpp
const Image& dataImage;
```

### Optional Mask

```cpp
const uint8_t* validMask;
```

Mask：

```text
0 = invalid
1 = valid
```

可以用于排除：

- Background
- ROI Edge
- Bad Pixel
- Saturated Pixel
- Known Defect
- Alignment Invalid Region

---

# 5. Output

支持：

```text
CV_8U
CV_16U
CV_32F
```

典型工业检测输出：

```text
Normalized Template
Normalized Data
Gain
Offset
Statistics
Valid Pixel Count
Clipping Count
```

推荐内部计算：

```text
CV_32F
```

最终根据后续算法需要转换：

```text
CV_32F → CV_8U
```

---

# 6. Overall Pipeline

```text
Template ROI
      │
      ├──────────────┐
      │              │
      ▼              ▼
Statistics       Statistics
      │              │
      ▼              ▼
Template Params   Data Params
      │              │
      └──────┬───────┘
             ▼
      Parameter Estimation
             │
       Gain / Offset
             │
             ▼
       Normalize Kernel
             │
             ▼
       Clipping / Mask
             │
             ▼
       Normalized Data
             │
             ▼
       Die-to-Die Difference
```

---

# 7. Strategy 1: Offset Normalize

## 7.1 Formula

```text
I' = I + Offset
```

其中：

```text
Offset = MeanT - MeanD
```

## 7.2 Purpose

只解决：

```text
整体亮度偏移
```

## 7.3 Example

```text
Template Mean = 100
Data Mean     = 120

Offset = 100 - 120
       = -20
```

因此：

```text
Data' = Data - 20
```

## 7.4 Advantage

- 极低计算量
- 非常适合 CUDA
- 不改变局部对比度
- 不容易破坏缺陷

## 7.5 Disadvantage

不能解决：

```text
Contrast Difference
```

---

# 8. Strategy 2: Gain Normalize

## 8.1 Formula

```text
I' = Gain × I
```

## 8.2 Gain

通常：

```text
Gain = StdT / StdD
```

## 8.3 Purpose

解决：

```text
对比度 / 动态范围差异
```

---

# 9. Strategy 3: Gain + Offset Normalize

推荐作为基础生产算法。

## 9.1 Formula

```text
I' = Gain × I + Offset
```

## 9.2 Parameters

```text
Gain = StdT / StdD

Offset = MeanT - Gain × MeanD
```

等价于：

```text
I' =
(I - MeanD)
×
StdT / StdD
+
MeanT
```

## 9.3 Objective

满足：

```text
Mean(I') ≈ MeanT
Std(I')  ≈ StdT
```

## 9.4 Protection

当：

```text
StdD < epsilon
```

禁止：

```text
Gain = StdT / StdD
```

必须：

```text
Gain = 1
```

或者退化到：

```text
Offset Normalize
```

---

# 10. Strategy 4: Mean Normalize

只进行均值对齐。

```text
I' = I - MeanD + MeanT
```

适合：

```text
亮度变化明显
对比度基本稳定
```

---

# 11. Strategy 5: Mean / StdDev Normalize

## 11.1 Standard Z-Score

```text
Z = (I - Mean) / StdDev
```

## 11.2 Template Matching Version

```text
I' =
(I - MeanD)
×
StdT / StdD
+
MeanT
```

这是最推荐的全局统计 Normalize。

---

# 12. Strategy 6: Min-Max Normalize

## 12.1 Formula

```text
I' =
(I - Min)
/
(Max - Min)
```

映射到 0~255：

```text
I' =
255 × (I - Min)
/
(Max - Min)
```

## 12.2 Protection

```text
Max - Min < epsilon
```

时必须进入 fallback。

## 12.3 Problem

Min/Max 对异常值敏感：

```text
Hot Pixel
Dead Pixel
Noise
Defect
```

均可能改变结果。

因此生产环境优先使用 Percentile。

---

# 13. Strategy 7: Percentile Normalize

推荐工业实现。

使用：

```text
P_low
P_high
```

替代：

```text
Min
Max
```

例如：

```text
P1
P99
```

## Formula

```text
I' =
255 ×
(I - P1)
/
(P99 - P1)
```

然后：

```text
I < P1  → 0
I > P99 → 255
```

## Advantages

降低：

- Noise
- Hot Pixel
- Defect
- Outlier

对 Normalize 参数的影响。

---

# 14. Strategy 8: Histogram Normalize

首先计算：

```text
H[256]
```

然后进行：

```text
Histogram
 ↓
CDF
 ↓
Mapping
```

可以实现：

```text
Histogram Equalization
```

但对于 Die-to-Die，更推荐 Histogram Matching。

---

# 15. Strategy 9: Histogram Matching

目标：

```text
CDFData
→
CDFTemplate
```

建立：

```text
LUT[256]
```

最终：

```text
I' = LUT[I]
```

Pipeline：

```text
Data
 ↓
Histogram
 ↓
CDF
 ↓
Template CDF
 ↓
LUT
 ↓
Pixel Mapping
```

## CUDA

推荐：

```text
Kernel 1:
Histogram

Kernel 2:
CDF / LUT

Kernel 3:
LUT Mapping
```

对于 `CV_8U` 图像，LUT Mapping 非常适合 GPU。

---

# 16. Strategy 10: Robust Mean

普通 Mean：

```text
Mean = ΣI / N
```

容易受到异常像素影响。

可选：

```text
Median
Trimmed Mean
Winsorized Mean
```

## Trimmed Mean

例如：

```text
排序
 ↓
去掉最低 5%
 ↓
去掉最高 5%
 ↓
Mean
```

适合存在异常像素的工业图像。

---

# 17. Strategy 11: Robust StdDev

可以使用 MAD：

```text
MAD =
median(|I - Median|)
```

近似：

```text
Std ≈ 1.4826 × MAD
```

比普通 StdDev 更抗异常值。

---

# 18. Strategy 12: Background Subtraction

估计背景：

```text
B(x,y) = Blur(I)
```

然后：

```text
I' = I - B
```

可使用：

```text
Gaussian
Box
Median
Morphological Opening
```

适合：

```text
非均匀照明
渐晕
大尺度背景
```

---

# 19. Strategy 13: Background Division

公式：

```text
I' = I / B
```

必须保护：

```text
B < epsilon
```

否则禁止除法。

Division Normalize 特别适合：

```text
乘性光照变化
```

例如：

```text
Observed = Object × Illumination
```

---

# 20. Strategy 14: Local Mean Normalize

局部窗口：

```text
W(x,y)
```

计算：

```text
μ(x,y)
```

然后：

```text
I' = I - μ(x,y)
```

适合：

```text
空间亮度变化
```

---

# 21. Strategy 15: Local Mean / Std Normalize

计算：

```text
μ(x,y)
σ(x,y)
```

公式：

```text
I' =
(I - μ(x,y))
/
(σ(x,y) + epsilon)
```

可以进一步映射到 Template：

```text
I' =
(I - μD(x,y))
×
σT(x,y) / σD(x,y)
+
μT(x,y)
```

这是计算量较大的高级方案。

---

# 22. Strategy 16: Gamma Normalize

公式：

```text
I' =
255 × (I / 255)^gamma
```

其中：

```text
gamma < 1
```

增强暗部。

```text
gamma > 1
```

压制暗部。

不建议作为默认 Die-to-Die Normalize。

---

# 23. Strategy 17: Log Normalize

公式：

```text
I' = log(1 + I)
```

适合：

```text
大动态范围
```

不建议作为普通 8-bit Die ROI 的默认方案。

---

# 24. Strategy 18: Z-Score Normalize

输出：

```text
CV_32F
```

公式：

```text
Z =
(I - Mean)
/
StdDev
```

适合：

```text
Correlation
Statistical Detection
Similarity
```

---

# 25. Strategy 19: Masked Normalize

所有统计量必须支持 Mask。

统计：

```text
Mean =
Σ(I × Mask)
/
ΣMask
```

只允许：

```text
Mask != 0
```

参与：

- Mean
- StdDev
- Min
- Max
- Histogram
- Percentile

---

# 26. Strategy 20: Defect Exclusion

如果已经存在 Defect Mask：

```text
ValidMask =
ROI Mask
AND
NOT DefectMask
```

然后：

```text
Statistics(ValidMask)
```

避免真实缺陷污染 Normalize 参数。

---

# 27. Strategy 21: Iterative Normalize

高级流程：

```text
First Normalize
      ↓
First Difference
      ↓
Initial Defect Mask
      ↓
Exclude Defect
      ↓
Recalculate Statistics
      ↓
Second Normalize
      ↓
Final Difference
```

默认：

```text
iteration = 1
```

生产环境最多允许有限迭代。

禁止无限迭代。

---

# 28. Recommended Strategy Priority

针对 Semiconductor Die-to-Die：

## Level 1

```text
Gain + Offset
+
Masked Statistics
+
Clipping
```

推荐作为默认方案。

## Level 2

```text
Percentile
+
Robust Statistics
```

用于异常值较多的场景。

## Level 3

```text
Background Correction
+
Gain + Offset
```

用于非均匀照明。

## Level 4

```text
Histogram Matching
```

用于非线性灰度分布变化。

## Level 5

```text
Local Mean / Std
```

用于强空间非均匀场景。

---

# 29. Recommended Default Algorithm

默认采用：

```text
Data ROI
   ↓
Valid Mask
   ↓
Robust Statistics
   ↓
Mean / Std
   ↓
Calculate Gain
   ↓
Calculate Offset
   ↓
CUDA Normalize
   ↓
Clamp
```

公式：

```text
Gain =
clamp(
    StdT / max(StdD, epsilon),
    GainMin,
    GainMax
)
```

```text
Offset =
MeanT -
Gain × MeanD
```

最终：

```text
I' =
clamp(
    Gain × I + Offset,
    OutputMin,
    OutputMax
)
```

---

# 30. Gain / Offset Safety Limits

必须支持：

```cpp
float gainMin;
float gainMax;

float offsetMin;
float offsetMax;
```

例如：

```text
Gain < gainMin
    → gainMin

Gain > gainMax
    → gainMax
```

防止统计异常导致：

```text
Gain = 0
Gain = 100
```

产生严重图像失真。

---

# 31. Saturation Handling

Normalize 后：

```text
I' < min
```

执行：

```text
I' = min
```

```text
I' > max
```

执行：

```text
I' = max
```

对于 `CV_8U`：

```text
[0,255]
```

必须统计：

```text
clipLowCount
clipHighCount
```

如果 Clipping 比例异常，应产生诊断信息。

---

# 32. Constant Image Handling

如果：

```text
StdD < epsilon
```

说明 Data ROI 几乎没有灰度变化。

不能直接：

```text
Gain = StdT / StdD
```

必须：

```text
Gain = 1
```

然后：

```text
Offset = MeanT - MeanD
```

或者直接使用：

```text
DefaultThreshold / fallback
```

具体策略由配置决定。

---

# 33. NaN / Inf Handling

`CV_32F` 必须检查：

```text
NaN
Inf
```

禁止进入：

```text
Gain
Offset
Normalize
Difference
```

所有统计结果必须验证：

```text
isfinite(mean)
isfinite(std)
isfinite(gain)
isfinite(offset)
```

---

# 34. CUDA Architecture

推荐 GPU Pipeline：

```text
GPU Memory
│
├── Template
│
├── Data
│
├── Valid Mask
│
├── Statistics Kernel
│
├── Parameter Kernel
│
├── Normalize Kernel
│
└── Difference Kernel
```

目标：

```text
GPU → GPU → GPU → GPU
```

禁止：

```text
GPU
 ↓
CPU
 ↓
GPU
```

除非确实需要 CPU 决策。

---

# 35. CUDA Kernel 1: Statistics

计算：

```text
Count
Sum
SumSquare
Min
Max
```

Mean：

```text
Mean = Sum / Count
```

Variance：

```text
Variance =
SumSquare / Count
-
Mean²
```

Std：

```text
Std = sqrt(max(Variance, 0))
```

推荐：

```text
Block Reduction
+
Warp Reduction
+
Global Reduction
```

---

# 36. CUDA Kernel 2: Parameter Calculation

输入：

```text
MeanT
StdT
MeanD
StdD
```

计算：

```text
Gain
Offset
```

可以放在：

```text
CUDA kernel
```

也可以放在：

```text
Host
```

如果参数计算非常少，Host 计算本身不是性能瓶颈。

但如果需要极低延迟 Pipeline：

```text
GPU Statistics
 ↓
GPU Parameters
 ↓
GPU Normalize
```

更合理。

---

# 37. CUDA Kernel 3: Normalize

核心：

```cpp
float value =
    gain * static_cast<float>(src[idx])
    + offset;

value = fminf(fmaxf(value, minValue), maxValue);

dst[idx] = static_cast<uint8_t>(value);
```

必须考虑：

```text
coalesced memory access
```

以及：

```text
vectorized load/store
```

---

# 38. CUDA Kernel Optimization

优先考虑：

```text
1. Coalesced Access
2. Minimal Global Memory Traffic
3. No unnecessary synchronization
4. Warp-level reduction
5. Persistent buffers
6. CUDA Streams
7. Pinned Host Memory
8. Async memcpy
9. Kernel Fusion
```

---

# 39. Kernel Fusion

可以考虑融合：

```text
Normalize
+
Clamp
+
Difference
```

即：

```text
src
 ↓
Normalize
 ↓
Clamp
 ↓
abs(Template - Data)
```

直接输出：

```text
Difference
```

避免写一次：

```text
Normalized Data
```

再重新读回来。

对于高吞吐 Die-to-Die，非常有价值。

---

# 40. NPP Mapping

可以优先使用 NVIDIA NPP 完成：

```text
Mean
StdDev
MinMax
Histogram
Add
Sub
Mul
AbsDiff
Filter
```

而将特殊逻辑放在：

```text
Custom CUDA Kernel
```

推荐：

```text
NPP
+
Custom CUDA
```

而不是所有操作全部自行实现。

---

# 41. CV-CUDA Mapping

CV-CUDA 可以作为：

```text
Image Pipeline
```

使用。

适合：

```text
Normalize
Filter
Resize
Warp
Color Conversion
```

但对于高度定制化的 Semiconductor Inspection Pipeline：

```text
Custom CUDA Kernel
```

通常拥有更大的优化空间。

---

# 42. CPU Reference Implementation

必须保留 CPU Reference。

目的：

```text
CPU
 ↓
Golden Result

CUDA
 ↓
GPU Result

Compare
```

比较：

```text
Max Error
Mean Error
RMSE
PSNR
Pixel Match
```

这样才能验证 CUDA 实现是否正确。

---

# 43. Numerical Accuracy

对于 `CV_8U` 输出：

允许：

```text
abs(CPU - CUDA) <= 1
```

对于 `CV_32F`：

建议：

```text
abs(CPU - CUDA)
<=
relative tolerance
```

所有测试必须固定：

```text
epsilon
rounding
clipping
```

避免 CPU/CUDA 因浮点舍入产生不确定结果。

---

# 44. Performance Requirements

性能指标：

```text
Latency
Throughput
GPU Utilization
Memory Bandwidth
Kernel Time
H2D
D2H
```

重点测量：

```text
Statistics
Normalize
Difference
```

不能只测试：

```text
Normalize Kernel
```

必须测试整个 Pipeline。

---

# 45. Benchmark Matrix

至少测试：

```text
256 × 256
512 × 512
1024 × 1024
2048 × 2048
4096 × 4096
```

数据类型：

```text
CV_8U
CV_16U
CV_32F
```

策略：

```text
Offset
Gain
GainOffset
MeanStd
Percentile
Histogram
Local
Background
```

---

# 46. Correctness Test Cases

必须测试：

## Case 1

```text
Template == Data
```

期望：

```text
Gain ≈ 1
Offset ≈ 0
Difference ≈ 0
```

## Case 2

```text
Data = Template + Constant
```

期望：

```text
Offset Correct
```

## Case 3

```text
Data = Gain × Template
```

期望：

```text
Gain Correct
```

## Case 4

```text
Data = Gain × Template + Offset
```

期望：

```text
Gain + Offset Correct
```

## Case 5

存在：

```text
Hot Pixel
```

检查 Robust Normalize。

## Case 6

存在：

```text
Defect
```

检查 Defect Mask。

## Case 7

```text
Std = 0
```

检查 fallback。

## Case 8

全部：

```text
255
```

检查 saturation。

---

# 47. Defect Preservation Test

这是本模块最重要的测试之一。

构造：

```text
Template
+
Artificial Defect
=
Data
```

经过 Normalize 后：

```text
Defect Contrast
```

必须仍然存在。

必须验证：

```text
Before Normalize Defect
vs
After Normalize Defect
```

不能只验证：

```text
Mean
Std
```

---

# 48. Diagnostic Output

建议输出：

```cpp
struct IntensityNormalizeResult
{
    float templateMean;
    float templateStd;

    float dataMean;
    float dataStd;

    float gain;
    float offset;

    float templateMin;
    float templateMax;

    float dataMin;
    float dataMax;

    uint64_t validPixelCount;

    uint64_t clipLowCount;
    uint64_t clipHighCount;

    bool fallbackUsed;

    int strategy;
};
```

---

# 49. Recommended API

```cpp
enum class IntensityNormalizeMethod
{
    Offset,
    Gain,
    GainOffset,
    MeanStd,
    MinMax,
    Percentile,
    HistogramMatching,
    Robust,
    BackgroundSubtract,
    BackgroundDivide,
    LocalMean,
    LocalMeanStd,
    Gamma,
    Log,
    ZScore
};
```

---

# 50. Configuration

```cpp
struct IntensityNormalizeParams
{
    IntensityNormalizeMethod method;

    float epsilon;

    float gainMin;
    float gainMax;

    float offsetMin;
    float offsetMax;

    float percentileLow;
    float percentileHigh;

    float gamma;

    int localRadius;

    float gaussianSigma;

    bool useMask;
    bool excludeDefect;
    bool enableClipping;

    float outputMin;
    float outputMax;
};
```

---

# 51. Recommended Class

```cpp
class IntensityNormalizer
{
public:

    IntensityNormalizer();

    explicit IntensityNormalizer(
        const IntensityNormalizeParams& params);

    bool normalize(
        const Image& templateImage,
        const Image& dataImage,
        Image& normalizedData,
        IntensityNormalizeResult& result);

    bool normalizeCuda(
        const GpuImage& templateImage,
        const GpuImage& dataImage,
        GpuImage& normalizedData,
        IntensityNormalizeResult& result,
        cudaStream_t stream);

private:

    IntensityNormalizeParams params_;
};
```

---

# 52. File Structure

推荐：

```text
IntensityNormalize/
│
├── IntensityNormalize.hpp
├── IntensityNormalize.cpp
│
├── IntensityNormalize.cuh
├── IntensityNormalize.cu
│
├── IntensityNormalizeStats.cuh
├── IntensityNormalizeStats.cu
│
├── IntensityNormalizeHistogram.cuh
├── IntensityNormalizeHistogram.cu
│
├── IntensityNormalizeBackground.cuh
├── IntensityNormalizeBackground.cu
│
└── tests/
    ├── test_cpu.cpp
    ├── test_cuda.cpp
    └── benchmark.cpp
```

---

# 53. Recommended Production Pipeline

针对当前 Die-to-Die 项目，默认推荐：

```text
Template ROI
       │
       ├── Valid Mask
       │
       ▼
Template Statistics
       │
       │
Data ROI
       │
       ├── Valid Mask
       │
       ▼
Data Statistics
       │
       ▼
Robust Validation
       │
       ▼
Gain / Offset
       │
       ▼
CUDA Normalize
       │
       ▼
Clamp
       │
       ▼
Normalized Data
       │
       ▼
Difference
       │
       ▼
Histogram Threshold
       │
       ▼
Defect Detection
```

---

# 54. Relationship With HistogramThreshold

`IntensityNormalize` 和 `HistogramThreshold` 必须保持独立。

```text
IntensityNormalize
=
灰度可比性

HistogramThreshold
=
灰度二值化阈值
```

推荐：

```text
Template
   │
   └──────────────┐
                  │
Data              │
 │                │
 ▼                ▼
Alignment → IntensityNormalize
                  │
                  ▼
              Difference
                  │
                  ▼
          HistogramThreshold
                  │
                  ▼
              Binary
```

---

# 55. GPU Data Residency Requirement

生产 Pipeline 中尽量保持：

```text
H2D
 ↓
GPU ROI
 ↓
Gray
 ↓
Alignment
 ↓
IntensityNormalize
 ↓
Difference
 ↓
Histogram
 ↓
Threshold
 ↓
Defect
 ↓
D2H
```

而不是：

```text
GPU
 ↓
CPU
 ↓
GPU
 ↓
CPU
```

---

# 56. Default Recommendation

第一版生产实现只实现：

```text
1. Offset
2. Gain
3. Gain + Offset
4. Mean / Std
5. Mask
6. Clipping
7. Robust Statistics
```

其中默认：

```text
Gain + Offset
+
Masked Mean/Std
+
Gain Limit
+
Offset Limit
+
Clipping
```

第二阶段：

```text
Percentile
Histogram Matching
Background Correction
```

第三阶段：

```text
Local Mean
Local Mean / Std
Iterative Normalize
```

---

# 57. Important Industrial Rule

禁止为了让：

```text
Template ≈ Data
```

而无限制 Normalize。

必须保证：

```text
Systematic Intensity Difference
        ↓
       Remove

Real Defect Signal
        ↓
       Preserve
```

因此任何新 Normalize 算法必须同时通过：

```text
1. Background suppression
2. False positive reduction
3. Defect preservation
4. CPU/GPU consistency
5. Performance benchmark
```

---

# 58. Final Algorithm Decision Tree

```text
是否存在明显整体亮度差？
        │
       Yes
        ↓
Offset / GainOffset

是否存在明显对比度差？
        │
       Yes
        ↓
GainOffset / MeanStd

是否存在异常像素？
        │
       Yes
        ↓
Percentile / Robust

是否存在非线性 Histogram 差异？
        │
       Yes
        ↓
Histogram Matching

是否存在空间亮度不均？
        │
       Yes
        ↓
Background Correction

是否存在明显局部亮度变化？
        │
       Yes
        ↓
Local Mean / Local Std
```

---

# 59. Final Recommended Architecture

```text
                    IntensityNormalize
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   Statistics           Background           Histogram
        │                   │                   │
 ┌──────┼──────┐            │             ┌────┴────┐
 │      │      │            │             │         │
Mean   Std   Robust       Sub/Divide    CDF       Matching
 │      │      │            │             │         │
 └──────┼──────┘            └──────┬──────┘         │
        │                          │                │
        └──────────────┬───────────┴────────────────┘
                       │
                Parameter Estimation
                       │
                  Gain / Offset
                       │
                 CUDA Normalize
                       │
                Clamp / Mask
                       │
                  Normalized Data
                       │
                  Difference
                       │
                Defect Detection
```

---

# 60. Implementation Priority

## P0 — 必须实现

```text
Gain + Offset
Mean / Std
Mask
Clipping
Fallback
CPU Reference
CUDA Kernel
Benchmark
```

## P1 — 强烈建议

```text
Percentile
Robust Statistics
Background Subtraction
Background Division
Histogram Matching
```

## P2 — 高级功能

```text
Local Mean
Local Mean / Std
Iterative Normalize
GPU Kernel Fusion
```

## P3 — 特殊场景

```text
Gamma
Log
```

---

# 61. Acceptance Criteria

模块只有满足以下条件才允许进入生产：

```text
[ ] CPU Reference 正确
[ ] CUDA 与 CPU 数值一致
[ ] Gain/Offset 异常有保护
[ ] Std=0 有 fallback
[ ] NaN/Inf 有保护
[ ] Mask 正确
[ ] Clipping 正确
[ ] Defect 不被明显削弱
[ ] 不产生明显新的 False Positive
[ ] GPU 无不必要同步
[ ] GPU Memory Access 合理
[ ] 支持 CUDA Stream
[ ] 支持异步 Pipeline
[ ] 完成不同 ROI Size Benchmark
[ ] 完成长期稳定性测试
```

---

# 62. Conclusion

`IntensityNormalize` 不是单一算法，而是一组用于建立：

```text
Template ↔ Data
```

灰度可比性的算法集合。

对于当前工业 Die-to-Die 检测，最推荐的核心实现为：

```text
Masked Statistics
        ↓
Mean / Std
        ↓
Gain + Offset
        ↓
Gain / Offset Limit
        ↓
CUDA Pixel Transform
        ↓
Clipping
        ↓
Difference
```

核心公式：

```text
Gain = StdT / max(StdD, epsilon)

Offset = MeanT - Gain × MeanD

NormalizedData =
clamp(
    Gain × Data + Offset,
    OutputMin,
    OutputMax
)
```

在此基础上逐步增加：

```text
Robust
Percentile
Histogram Matching
Background Correction
Local Normalization
```

最终形成完整的工业级 `IntensityNormalize` Engine。