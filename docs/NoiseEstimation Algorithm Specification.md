# NoiseEstimation Algorithm Specification

**Project:** Industrial D2D / SDI Inspection  
**Module:** Noise Estimation  
**Language:** C++ / CUDA  
**Target:** Industrial High-Speed Wafer / Die Inspection  
**Version:** 1.0  
**Status:** Implementation Specification

---

# 1. Purpose

`NoiseEstimation` 模块用于估计工业晶圆 / Die Inspection 图像中的背景噪声水平，并为后续：

- Difference Image
- Adaptive Threshold
- Defect Detection
- Blob Extraction
- False Positive Suppression
- D2D / SDI Detection

提供稳定的噪声尺度。

模块必须支持：

1. Global Noise Estimation
2. Local Noise Estimation
3. MAD Noise Estimation
4. Difference Noise Estimation
5. Auto Noise Model Selection
6. Auto Threshold
7. CPU Reference Implementation
8. CUDA 高性能实现
9. ROI / Mask 支持
10. 8-bit / 16-bit 输入
11. 多 Die / Tile 批量处理
12. 与现有 D2D Pipeline 无缝集成

---

# 2. Design Goals

## 2.1 Primary Goals

### G1. 稳定

对于正常无缺陷区域：

```text
Noise Estimate ≈ Background Noise σ
```

不能因为少量缺陷导致噪声估计严重偏大。

---

### G2. Robust

必须抑制：

- Defect
- Edge
- Pattern
- Bright Pixel
- Dark Pixel
- Hot Pixel
- Dead Pixel
- Registration Residual

对 Noise Estimate 的影响。

---

### G3. High Performance

CUDA 版本必须支持：

```text
large ROI
large wafer
multi-die
batch processing
```

并避免：

- CPU/GPU 来回拷贝
- 不必要的 global memory
- 多次 kernel launch
- atomic contention
- 重复 histogram calculation

---

# 3. Overall Detection Pipeline

推荐完整 D2D Pipeline：

```text
Template ROI
      │
      ▼
Data ROI
      │
      ├──────────────┐
      │              │
      ▼              ▼
Gray Template    Gray Data
      │              │
      ▼              ▼
Intensity Normalize
      │              │
      └──────┬───────┘
             ▼
       Image Alignment
      Phase Correlation
             │
             ▼
          Warp Data
             │
             ▼
       Difference Image
             │
             ▼
      Noise Estimation
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
   Global   Local     MAD
     │       │        │
     └───────┼────────┘
             ▼
       Noise Fusion
             │
             ▼
      Auto Threshold
             │
             ▼
      Binary Defect Map
             │
             ▼
      Morphology
             │
             ▼
       Connected CC
             │
             ▼
       Defect Objects
             │
             ▼
      Defect Filtering
             │
             ▼
      Final Detection
```

---

# 4. Noise Definition

对于图像：

```text
I(x,y)
```

定义：

```text
I(x,y) = S(x,y) + N(x,y)
```

其中：

```text
S(x,y) = Signal
N(x,y) = Noise
```

Noise Estimation 的目标：

```text
estimate σN
```

即：

```text
σN = StandardDeviation(N)
```

---

# 5. Noise Sources

D2D / SDI 中 Noise 可能来自：

```text
Camera Noise
Photon Noise
Read Noise
ADC Noise
Illumination Noise
Pixel Response Non-Uniformity
Pattern Residual
Alignment Error
Focus Variation
Wafer Surface Variation
Compression / Quantization
```

实际工程中不能简单：

```cpp
stddev(image)
```

因为：

```text
stddev(image)
=
Signal Variation
+
Pattern Variation
+
Defect
+
Noise
```

所以必须使用 Robust Estimator。

---

# 6. Noise Estimation Methods

模块至少实现四种方法：

```text
GLOBAL_STD
LOCAL_STD
MAD
DIFFERENCE_NOISE
```

---

# 7. Global Noise Estimation

## 7.1 Definition

对 ROI 中所有有效像素计算：

```text
μ = mean(I)
```

```text
σ² = mean((I - μ)²)
```

```text
σ = sqrt(σ²)
```

---

## 7.2 Formula

对于 N 个有效像素：

```text
μ = Σ Xi / N
```

```text
σ² = Σ Xi² / N - μ²
```

```text
σ = sqrt(max(σ², 0))
```

---

## 7.3 Limitation

Global STD 对以下情况敏感：

```text
Defect
Strong Edge
Pattern
Large Bright Region
Large Dark Region
```

因此 Global STD：

```text
只作为 baseline
```

不能作为唯一生产级 Noise Estimate。

---

# 8. Robust Global Noise Estimation

推荐：

```text
Median + MAD
```

---

# 9. MAD Noise Estimation

MAD：

```text
MAD = median(|Xi - median(X)|)
```

对于 Gaussian Noise：

```text
σ ≈ 1.4826 × MAD
```

因此：

```text
sigmaMAD = 1.4826 * MAD
```

---

# 10. Why MAD

MAD 对 outlier 非常稳定。

例如：

```text
Normal Pixels:
100 101 99 100 102 98 101 ...

Defect:
255
```

STD：

```text
明显增加
```

MAD：

```text
基本不受影响
```

因此：

```text
MAD
```

作为 Robust Noise Estimator。

---

# 11. Local Noise Estimation

Global Noise 无法描述空间变化。

因此支持 Local Noise Map：

```text
σ(x,y)
```

将 ROI 分成：

```text
Tile / Window
```

例如：

```text
32 × 32
64 × 64
128 × 128
```

对每个 Window 计算：

```text
localMean
localStd
```

得到：

```text
NoiseMap
```

---

# 12. Local Noise Formula

对于 window W：

```text
μW = mean(W)
```

```text
σW = sqrt(E[X²] - E[X]²)
```

但是生产环境必须增加 robust filtering。

推荐：

```text
σW = MAD(W) × 1.4826
```

或者：

```text
σW = min(
    STD(W),
    1.4826 × MAD(W)
)
```

---

# 13. Difference Noise

D2D 中最重要的 Noise Estimate 之一来自 Difference Image。

定义：

```text
D(x,y) = Data(x,y) - Template(x,y)
```

如果：

```text
Data = S + N1
Template = S + N2
```

则：

```text
D = N1 - N2
```

如果两幅图噪声独立：

```text
σD² = σ1² + σ2²
```

若两幅图噪声相同：

```text
σ1 = σ2 = σ
```

则：

```text
σD = sqrt(2) × σ
```

所以：

```text
σsingle ≈ σD / sqrt(2)
```

---

# 14. Difference Noise Estimation

推荐：

```text
D = abs(Data - Template)
```

但是 Noise Estimation 时不要直接使用：

```text
abs(D)
```

而应该保留 signed difference：

```text
D = Data - Template
```

计算：

```text
median(D)
MAD(D)
```

最终：

```text
sigmaDiff = 1.4826 × MAD(D) / sqrt(2)
```

---

# 15. Difference Noise 优势

对于 D2D：

```text
Template
    +
Data
    ↓
Difference
```

Difference Noise 天然包含：

```text
Camera Noise
Registration Residual
Illumination Residual
Template/Data variation
```

因此它比单图 Noise 更贴近最终检测噪声。

---

# 16. Registration Residual

必须考虑 alignment residual。

假设：

```text
D = Data - Warp(Template)
```

那么：

```text
D = Noise + RegistrationResidual + Defect
```

因此：

```text
σDiff
```

实际上代表：

```text
Effective Detection Noise
```

推荐将其作为最终 threshold 的主要输入。

---

# 17. Mask

Noise Estimation 必须支持 Mask。

Mask：

```text
0 = ignore
1 = valid
```

必须排除：

```text
Wafer Edge
Die Edge
Alignment Border
Known Defect
Invalid Pixel
Saturated Pixel
Background
```

API：

```cpp
NoiseEstimate(
    image,
    mask,
    params
)
```

---

# 18. Saturation Handling

8-bit：

```text
0
255
```

16-bit：

```text
0
65535
```

可配置：

```cpp
ignoreSaturated = true;
```

推荐默认：

```text
ignore zero
ignore max
```

避免：

```text
dead pixel
saturated pixel
```

影响 Noise。

---

# 19. Histogram Based Noise Estimation

对于 8-bit 图像：

```text
Histogram[256]
```

可以计算：

```text
Median
Percentile
MAD approximation
```

推荐：

```text
Histogram-based median
```

避免完整排序。

---

# 20. Percentile

支持：

```text
P01
P05
P25
P50
P75
P95
P99
```

其中：

```text
Median = P50
```

---

# 21. Histogram MAD

步骤：

```text
1. Histogram
2. Find Median
3. Calculate |X - Median|
4. Histogram absolute deviation
5. Find median deviation
6. MAD = medianDeviation
7. sigma = 1.4826 × MAD
```

CUDA 可以使用：

```text
256-bin histogram
```

完成。

---

# 22. Auto Noise Estimation

生产环境推荐最终接口：

```cpp
AutoNoiseEstimate()
```

自动融合：

```text
Global STD
MAD
Difference Noise
Local Noise
```

---

# 23. Noise Selection Strategy

推荐优先级：

```text
Difference MAD
        ↓
Global MAD
        ↓
Local MAD
        ↓
Global STD
```

但是不能简单使用第一个结果。

推荐：

```text
sigma_final =
max(
    sigma_diff,
    sigma_local
)
```

并限制异常值：

```text
sigma_min <= sigma_final <= sigma_max
```

---

# 24. Recommended Production Formula

对于 D2D：

```text
sigma_diff =
1.4826 × MAD(Data - Template) / sqrt(2)
```

然后：

```text
sigma_local =
median(LocalSigmaMap)
```

最终：

```text
sigma_noise =
max(
    sigma_diff,
    sigma_local
)
```

---

# 25. Local Noise Map

定义：

```cpp
cv::Mat noiseMap;
```

每个 tile 一个值：

```text
noiseMap(i,j) = sigma_local
```

为了生成 pixel-level map：

```text
Bilinear Interpolation
```

或：

```text
Nearest
```

---

# 26. Noise Map Smoothing

Local Noise Map 必须进行平滑。

推荐：

```text
Median Filter
```

例如：

```text
3 × 3
```

然后：

```text
Gaussian smoothing
```

可选。

---

# 27. Auto Threshold

Noise Estimate 最终用于：

```text
Threshold
```

基本公式：

```text
T = K × σ
```

例如：

```text
K = 4
5
6
7
8
```

---

# 28. Two-Sided Threshold

Difference：

```text
D = Data - Template
```

使用：

```text
abs(D) > T
```

其中：

```text
T = K × sigma_noise
```

---

# 29. Mean Bias Correction

Difference 可能存在 DC Offset：

```text
median(D) != 0
```

因此：

```text
Dcorrected = D - median(D)
```

然后：

```text
MAD(Dcorrected)
```

推荐生产实现：

```text
bias = median(D)
```

```text
D' = D - bias
```

---

# 30. Robust Threshold

推荐：

```text
T = bias + K × sigma
```

对于 absolute difference：

```text
abs(D - bias) > K × sigma
```

---

# 31. Multi-Level Threshold

支持：

```text
T1 = 4σ
T2 = 5σ
T3 = 6σ
T4 = 8σ
```

用于：

```text
Weak Defect
Medium Defect
Strong Defect
Critical Defect
```

---

# 32. Adaptive Local Threshold

对于 Local Noise：

```text
T(x,y) = K × σ(x,y)
```

检测：

```text
abs(D(x,y) - bias) > T(x,y)
```

这样可以处理：

```text
low-noise region
high-noise region
illumination variation
```

---

# 33. Noise Confidence

输出：

```cpp
float confidence;
```

推荐定义：

```text
confidence =
1 - uncertainty / sigma
```

或者使用：

```text
MAD / STD
```

判断 Gaussian 程度。

例如：

```text
ratio = sigmaMAD / sigmaSTD
```

正常 Gaussian：

```text
ratio ≈ 1
```

异常：

```text
ratio << 1
```

说明 STD 被 outlier 拉高。

---

# 34. Noise Result

统一结果：

```cpp
struct NoiseEstimateResult
{
    float globalMean;
    float globalStd;

    float median;
    float mad;

    float sigmaMAD;

    float differenceMean;
    float differenceStd;

    float differenceMedian;
    float differenceMAD;

    float sigmaDifference;

    float localSigmaMedian;
    float localSigmaMean;
    float localSigmaMin;
    float localSigmaMax;

    float finalSigma;

    float threshold;

    float confidence;

    int validPixelCount;

    bool valid;
};
```

---

# 35. Noise Parameters

```cpp
struct NoiseEstimateParams
{
    NoiseMethod method;

    int localWindowWidth;
    int localWindowHeight;

    float thresholdK;

    float sigmaMin;
    float sigmaMax;

    bool useMask;
    bool ignoreZero;
    bool ignoreSaturated;

    bool useDifference;
    bool useLocalNoise;
    bool useMAD;

    bool removeBias;

    int histogramBins;
};
```

---

# 36. NoiseMethod

```cpp
enum class NoiseMethod
{
    GLOBAL_STD,
    GLOBAL_MAD,
    LOCAL_STD,
    LOCAL_MAD,
    DIFFERENCE_STD,
    DIFFERENCE_MAD,
    AUTO
};
```

---

# 37. CPU API

推荐：

```cpp
NoiseEstimateResult estimateNoise(
    const cv::Mat& image,
    const cv::Mat& mask,
    const NoiseEstimateParams& params
);
```

D2D：

```cpp
NoiseEstimateResult estimateDifferenceNoise(
    const cv::Mat& data,
    const cv::Mat& templ,
    const cv::Mat& mask,
    const NoiseEstimateParams& params
);
```

Auto：

```cpp
NoiseEstimateResult autoNoiseEstimate(
    const cv::Mat& data,
    const cv::Mat& templ,
    const cv::Mat& mask,
    const NoiseEstimateParams& params
);
```

---

# 38. CUDA API

推荐：

```cpp
NoiseEstimateResult estimateNoiseCuda(
    const cv::cuda::GpuMat& image,
    const cv::cuda::GpuMat& mask,
    const NoiseEstimateParams& params
);
```

D2D：

```cpp
NoiseEstimateResult estimateDifferenceNoiseCuda(
    const cv::cuda::GpuMat& data,
    const cv::cuda::GpuMat& templ,
    const cv::cuda::GpuMat& mask,
    const NoiseEstimateParams& params
);
```

---

# 39. CUDA Pipeline

推荐：

```text
Input GpuMat
    │
    ▼
Kernel: Difference
    │
    ▼
Kernel: Mask / Valid Pixel
    │
    ├───────────────┐
    ▼               ▼
Histogram       Sum/SumSq
    │               │
    ▼               ▼
Median           Mean/STD
    │
    ▼
MAD Histogram
    │
    ▼
MAD
    │
    ▼
Sigma
    │
    ▼
Threshold
```

---

# 40. CUDA Histogram

8-bit：

```cpp
__global__
void histogram256Kernel(
    const uint8_t* image,
    const uint8_t* mask,
    int width,
    int height,
    unsigned int* histogram
);
```

推荐：

```text
per-block shared histogram
```

然后：

```text
global reduction
```

避免大量：

```cpp
atomicAdd(globalHistogram + bin, 1);
```

---

# 41. Shared Histogram

每个 CUDA block：

```text
256 bins
```

流程：

```text
sharedHistogram[256] = 0

thread:
    atomicAdd(sharedHistogram + pixel, 1)

__syncthreads()

thread:
    atomicAdd(globalHistogram + bin,
              sharedHistogram[bin])
```

---

# 42. Warp Optimization

CUDA 版本优先：

```text
warp-level reduction
```

减少：

```text
atomicAdd
```

推荐：

```text
warp histogram
→ block histogram
→ global histogram
```

---

# 43. Mean / Variance

不要：

```text
每个 pixel atomicAdd global sum
```

推荐：

```text
Thread
 ↓
Warp reduction
 ↓
Block reduction
 ↓
Global result
```

统计：

```text
sum
sumSq
count
```

---

# 44. Numerically Stable Variance

如果使用：

```text
E[X²] - E[X]²
```

必须：

```cpp
double
```

或 CUDA：

```cpp
double
```

进行 accumulation。

避免：

```text
large ROI
large intensity
```

导致 precision error。

---

# 45. CUDA Difference Kernel

推荐：

```cpp
__global__
void differenceKernel(
    const uint8_t* data,
    const uint8_t* templ,
    int16_t* diff,
    int width,
    int height
);
```

不要直接：

```cpp
uint8_t diff
```

因为：

```text
Data - Template
```

需要 signed。

---

# 46. Difference Data Type

推荐：

```text
8-bit input
→ int16 difference
```

范围：

```text
[-255,255]
```

16-bit：

```text
→ int32
```

范围：

```text
[-65535,65535]
```

---

# 47. Absolute Difference

仅在最终 threshold：

```cpp
abs(diff)
```

Noise Estimation 阶段保留：

```text
signed difference
```

用于：

```text
bias estimation
MAD
```

---

# 48. Bias Estimation

```text
bias = median(diff)
```

然后：

```text
diffCorrected = diff - bias
```

---

# 49. MAD CUDA Implementation

对于 8-bit absolute deviation：

```text
MAD range:
0~255
```

因此：

```text
256-bin histogram
```

非常适合 CUDA。

流程：

```text
difference
 ↓
median
 ↓
abs(diff - median)
 ↓
histogram256
 ↓
median deviation
 ↓
MAD
 ↓
1.4826 × MAD
```

---

# 50. Difference Noise

最终：

```cpp
sigmaDifference =
1.4826f * MAD / sqrtf(2.0f);
```

---

# 51. Important: Difference Noise Scaling

如果 Template 与 Data 使用相同 acquisition noise：

```text
sigmaData = sigmaTemplate
```

则：

```text
sigmaDifference = sqrt(2) sigma
```

因此：

```text
sigma = sigmaDifference / sqrt(2)
```

如果两幅图 Noise 不同：

```text
sigmaDifference² =
sigmaData² + sigmaTemplate²
```

此时不要强制除：

```text
sqrt(2)
```

而应该输出：

```text
sigmaDifference
```

作为 detection noise。

---

# 52. Production Recommendation

D2D Detection Threshold：

```text
T = K × sigmaDifference
```

不要一定转换成 single-image sigma。

因为：

```text
最终检测对象就是 Difference Image
```

因此推荐：

```text
Threshold = K × sigmaDifference
```

---

# 53. Local Noise CUDA

每个 tile：

```text
64 × 64
```

计算：

```text
local histogram
```

得到：

```text
local median
local MAD
local sigma
```

输出：

```text
NoiseMap
```

---

# 54. Tile Size

默认：

```text
64 × 64
```

可选：

```text
32 × 32
64 × 64
128 × 128
```

推荐：

```text
64 × 64
```

原因：

```text
Noise statistics sufficient
CUDA block friendly
memory access efficient
```

---

# 55. Edge Handling

Local window 在 ROI 边界：

```text
partial tile
```

必须支持。

不要：

```text
越界访问
```

推荐：

```text
valid pixel count
```

低于：

```text
minValidPixels
```

则：

```text
localSigma = NaN
```

之后使用：

```text
globalSigma
```

fallback。

---

# 56. Noise Fusion

最终：

```cpp
float sigma = sigmaDifference;
```

如果 Local Noise 有效：

```cpp
sigma = max(
    sigmaDifference,
    localSigmaMedian
);
```

也可以：

```text
sigma =
alpha * sigmaDifference
+
(1-alpha) * sigmaLocal
```

默认不推荐平均。

---

# 57. Conservative Strategy

工业检测更推荐：

```text
max()
```

原因：

```text
避免 underestimated noise
```

因为：

```text
低估 noise
→ false positive
```

---

# 58. Sigma Clamp

必须：

```cpp
sigma = clamp(
    sigma,
    sigmaMin,
    sigmaMax
);
```

默认建议：

```text
sigmaMin = 0.5
sigmaMax = 50
```

具体值必须由 Camera / illumination / image bit depth 标定。

---

# 59. Auto Threshold

默认：

```text
K = 5
```

即：

```text
T = 5 × sigma
```

生产参数必须可配置。

---

# 60. Threshold Presets

建议：

```text
LOW:
K = 3.5

NORMAL:
K = 5

HIGH:
K = 6

VERY_HIGH:
K = 8
```

---

# 61. Defect Detection

最终：

```cpp
binary(x,y) =
    abs(diff(x,y) - bias) > threshold;
```

即：

```text
abs(D - bias) > Kσ
```

---

# 62. Local Threshold

如果启用 Local Noise：

```cpp
threshold(x,y) =
K * noiseMap(x,y);
```

检测：

```cpp
abs(diff(x,y) - bias)
    >
K * noiseMap(x,y);
```

---

# 63. Noise-Based Confidence

定义：

```text
SNR(x,y) =
abs(diff(x,y) - bias) / sigma
```

则：

```text
SNR < 3
```

通常认为：

```text
Noise
```

```text
3 <= SNR < 5
```

可能：

```text
Weak candidate
```

```text
SNR >= 5
```

强 defect candidate。

---

# 64. Candidate Classification

建议：

```text
SNR < 3
    Background

3 <= SNR < 5
    Weak

5 <= SNR < 8
    Defect

SNR >= 8
    Strong Defect
```

这些数值必须允许配置。

---

# 65. ROI Strategy

Noise Estimation 不应该使用整个 Wafer。

推荐：

```text
Die ROI
```

或者：

```text
Valid Inspection ROI
```

优先排除：

```text
Die Boundary
Wafer Edge
Alignment Region
Known Pattern Edge
```

---

# 66. D2D Recommendation

推荐 Noise ROI：

```text
Template ROI
Data ROI
      ↓
Alignment
      ↓
Common Valid ROI
      ↓
Difference
      ↓
Noise Estimation
```

Noise Estimate 必须基于：

```text
Common Valid ROI
```

---

# 67. SDI Recommendation

Single Die 没有 Template Die。

因此 Noise Estimation 推荐：

```text
Local background
+
Flat Region
+
MAD
+
High-pass residual
```

可以使用：

```text
D = I - GaussianBlur(I)
```

然后：

```text
sigma = 1.4826 × MAD(D)
```

---

# 68. SDI High-Pass Noise

推荐：

```text
Blur = Gaussian(I, sigma=1~3)
```

```text
Residual = I - Blur
```

然后：

```text
sigmaNoise =
1.4826 × MAD(Residual)
```

---

# 69. D2D vs SDI

| Method | D2D | SDI |
|---|---:|---:|
| Global STD | ✓ | ✓ |
| Global MAD | ✓ | ✓ |
| Local MAD | ✓ | ✓ |
| Difference MAD | ★★★★★ | - |
| High-pass MAD | ✓ | ★★★★★ |
| Local Noise Map | ★★★★ | ★★★★★ |

---

# 70. API Architecture

建议：

```text
NoiseEstimation/
│
├── include/
│   ├── NoiseEstimation.h
│   ├── NoiseTypes.h
│   └── NoiseCuda.h
│
├── src/
│   ├── NoiseEstimation.cpp
│   ├── GlobalNoise.cpp
│   ├── LocalNoise.cpp
│   ├── MADNoise.cpp
│   └── DifferenceNoise.cpp
│
├── cuda/
│   ├── NoiseHistogram.cu
│   ├── NoiseReduction.cu
│   ├── DifferenceNoise.cu
│   └── LocalNoise.cu
│
└── tests/
    ├── test_global_noise.cpp
    ├── test_mad_noise.cpp
    ├── test_difference_noise.cpp
    └── test_cuda_noise.cpp
```

---

# 71. C++ Class

推荐：

```cpp
class NoiseEstimator
{
public:

    NoiseEstimateResult estimate(
        const cv::Mat& image,
        const cv::Mat& mask,
        const NoiseEstimateParams& params
    );

    NoiseEstimateResult estimateDifference(
        const cv::Mat& data,
        const cv::Mat& templ,
        const cv::Mat& mask,
        const NoiseEstimateParams& params
    );

    NoiseEstimateResult estimateSDI(
        const cv::Mat& image,
        const cv::Mat& mask,
        const NoiseEstimateParams& params
    );
};
```

---

# 72. CUDA Class

```cpp
class CudaNoiseEstimator
{
public:

    NoiseEstimateResult estimate(
        const cv::cuda::GpuMat& image,
        const cv::cuda::GpuMat& mask,
        const NoiseEstimateParams& params
    );

    NoiseEstimateResult estimateDifference(
        const cv::cuda::GpuMat& data,
        const cv::cuda::GpuMat& templ,
        const cv::cuda::GpuMat& mask,
        const NoiseEstimateParams& params
    );
};
```

---

# 73. Memory Management

禁止：

```text
每个 ROI malloc/free
```

必须：

```text
Persistent CUDA Workspace
```

例如：

```cpp
struct NoiseCudaWorkspace
{
    uint32_t* histogram;
    uint32_t* localHistogram;

    double* sum;
    double* sumSq;

    uint32_t* count;

    int16_t* difference;

    float* noiseMap;
};
```

---

# 74. Workspace Reuse

推荐：

```cpp
workspace.resize(width, height);
```

只在尺寸发生变化时重新分配。

---

# 75. CUDA Stream

所有 CUDA 操作：

```cpp
cudaStream_t stream
```

支持异步：

```text
H2D
Kernel
Reduction
D2H
```

---

# 76. Avoid Synchronization

禁止：

```cpp
cudaDeviceSynchronize()
```

在每个 kernel 后调用。

推荐：

```text
same CUDA stream
```

最后：

```text
one synchronization
```

---

# 77. CPU/GPU Data Transfer

推荐：

```text
Image
 ↓
GPU
 ↓
Alignment
 ↓
Difference
 ↓
Noise
 ↓
Threshold
 ↓
Binary
 ↓
Blob
```

整个 Pipeline 尽可能保持在 GPU。

不要：

```text
GPU Difference
 ↓
CPU Noise
 ↓
GPU Threshold
```

---

# 78. Performance Target

对于：

```text
4096 × 4096
8-bit
```

目标：

```text
Noise estimation < 1 ms
```

理想：

```text
< 0.5 ms
```

具体取决于 GPU。

---

# 79. Benchmark Metrics

必须记录：

```text
Image Size
Pixel Count
CPU Time
GPU Time
H2D Time
D2H Time
Kernel Time
Memory Bandwidth
Kernel Launch Count
```

---

# 80. Performance Test

测试：

```text
1024 × 1024
2048 × 2048
4096 × 4096
8192 × 8192
```

记录：

```text
ms
MPixel/s
GB/s
```

---

# 81. Correctness Tests

必须验证：

### Test 1

纯 Gaussian Noise：

```text
σ = 2
```

估计：

```text
≈ 2
```

---

### Test 2

加入 1% Outlier：

```text
1% pixels = 255
```

MAD 应保持稳定。

---

### Test 3

加入 Defect：

```text
large bright defect
```

Noise Estimate 不应明显增加。

---

### Test 4

Data / Template：

```text
σ1 = 2
σ2 = 2
```

理论：

```text
σDifference = sqrt(8)
             ≈ 2.828
```

如果转换 single-image sigma：

```text
2.828 / sqrt(2)
≈ 2
```

---

# 82. CUDA vs CPU Validation

要求：

```text
abs(cpuSigma - cudaSigma)
    < tolerance
```

建议：

```text
8-bit:
absolute error <= 0.1

16-bit:
relative error <= 0.1%
```

---

# 83. Determinism

CUDA Histogram + Reduction 必须尽可能保持：

```text
deterministic
```

允许：

```text
floating point reduction
```

产生微小误差。

但不能出现：

```text
threshold randomly changes
```

---

# 84. Error Handling

输入为空：

```cpp
if (image.empty())
    return invalid;
```

尺寸不同：

```cpp
if (data.size() != templ.size())
    return invalid;
```

Mask 尺寸不同：

```cpp
return invalid;
```

有效像素过少：

```cpp
valid = false;
```

---

# 85. Minimum Valid Pixels

推荐：

```text
minValidPixels =
max(1024, width × height × 0.01)
```

如果：

```text
validPixelCount < minValidPixels
```

则：

```text
fallback
```

---

# 86. Fallback Strategy

推荐：

```text
Difference MAD
      ↓ fail
Global MAD
      ↓ fail
Global STD
      ↓ fail
Configured Default Sigma
```

默认：

```text
sigma = 2.0
```

但该默认值必须允许 Camera Profile 配置。

---

# 87. Camera Profile

不同相机：

```text
camera
illumination
gain
exposure
bit depth
```

Noise 不一样。

建议配置：

```json
{
    "camera": "CameraA",
    "bitDepth": 8,
    "sigmaMin": 0.5,
    "sigmaMax": 30.0,
    "thresholdK": 5.0,
    "localWindow": 64
}
```

---

# 88. Debug Output

开发模式支持输出：

```text
noise.csv
noise_map.png
difference.png
histogram.csv
threshold.png
```

生产模式：

```text
disable debug output
```

---

# 89. Logging

输出：

```text
NoiseEstimation:
validPixels=1234567
median=101.3
MAD=1.7
sigmaMAD=2.52
sigmaDifference=2.01
localSigma=2.14
finalSigma=2.14
threshold=10.70
```

---

# 90. Recommended D2D Production Configuration

```cpp
NoiseEstimateParams params;

params.method = NoiseMethod::AUTO;

params.localWindowWidth  = 64;
params.localWindowHeight = 64;

params.thresholdK = 5.0f;

params.sigmaMin = 0.5f;
params.sigmaMax = 30.0f;

params.useMask = true;
params.ignoreZero = true;
params.ignoreSaturated = true;

params.useDifference = true;
params.useLocalNoise = true;
params.useMAD = true;

params.removeBias = true;

params.histogramBins = 256;
```

---

# 91. Recommended D2D Algorithm

最终推荐：

```text
Data
Template
   │
   ▼
Alignment
   │
   ▼
Warp
   │
   ▼
Difference
   │
   ▼
Valid Mask
   │
   ▼
Median(D)
   │
   ▼
D' = D - Median(D)
   │
   ▼
MAD(D')
   │
   ▼
sigmaDiff = 1.4826 × MAD / sqrt(2)
   │
   ├──────────────┐
   │              │
   ▼              ▼
Local MAD     Global MAD
   │              │
   └──────┬───────┘
          ▼
       Noise Fusion
          │
          ▼
       sigmaNoise
          │
          ▼
       T = K × σ
          │
          ▼
abs(D') > T
          │
          ▼
      Binary Map
```

---

# 92. Recommended SDI Algorithm

Single Die：

```text
Image
  │
  ▼
Normalize
  │
  ▼
Gaussian Blur
  │
  ▼
High Pass
  │
  ▼
Residual
  │
  ▼
Median
  │
  ▼
MAD
  │
  ▼
sigmaNoise
  │
  ▼
Adaptive Threshold
  │
  ▼
Defect Map
```

---

# 93. Important Engineering Rule

不要把 Noise Estimation 做成：

```cpp
float NoiseEstimation(cv::Mat image);
```

最终生产级接口必须保留：

```text
global statistics
local statistics
difference statistics
mask
confidence
threshold
fallback
```

否则后续无法支持：

```text
D2D
SDI
Adaptive Threshold
Recipe
Camera Calibration
Auto Tuning
```

---

# 94. Codex / Cursor Implementation Requirements

实现时必须：

### Requirement 1

先扫描现有工程。

不得直接创建重复：

```text
Histogram
Threshold
Normalize
Difference
CUDA utility
```

---

### Requirement 2

复用现有：

```text
Image
ROI
Mask
CUDA context
CUDA stream
Memory pool
```

---

### Requirement 3

CPU 是 Reference Implementation。

CUDA 必须：

```text
match CPU result
```

---

### Requirement 4

所有新增 CUDA kernel：

```text
.cu
```

所有接口：

```text
.h
```

禁止将大量 CUDA implementation 写进 header。

---

### Requirement 5

必须保持：

```text
C++17
```

除非现有工程已经使用更高版本。

---

### Requirement 6

不得修改现有 public API。

如必须修改：

```text
新增 overload
```

而不是破坏旧接口。

---

### Requirement 7

不得引入新的大型第三方依赖。

优先：

```text
OpenCV
CUDA Runtime
CUDA CUB
```

如果项目已经存在 CUB，则优先使用 CUB。

---

# 95. CUDA Optimization Priority

优化顺序：

```text
1. Reduce CPU/GPU transfer
2. Fuse kernels
3. Shared-memory histogram
4. Warp reduction
5. CUB reduction
6. Persistent workspace
7. Stream overlap
8. Memory coalescing
9. Avoid atomics
10. Kernel launch reduction
```

---

# 96. Kernel Fusion

如果性能不足，可以融合：

```text
Difference
+
Mask
+
AbsDeviation
```

但是：

```text
可维护性
```

优先于过度 Fusion。

---

# 97. Memory Access

必须确保：

```text
threadIdx.x
```

对应连续像素。

推荐：

```text
coalesced global memory access
```

避免：

```text
strided access
```

---

# 98. Avoid Full Noise Map When Not Needed

如果只需要：

```text
global sigma
```

不要生成：

```text
pixel-level noise map
```

减少：

```text
GPU memory
bandwidth
```

---

# 99. API Execution Modes

支持：

```cpp
NoiseMode::GLOBAL
NoiseMode::LOCAL
NoiseMode::DIFFERENCE
NoiseMode::AUTO
```

AUTO 为生产默认。

---

# 100. Final Production Recommendation

对于当前 D2D 工程，默认采用：

```text
Difference-based MAD
+
Local MAD
+
Robust Bias Removal
+
Adaptive Threshold
```

核心公式：

```text
D = Data - Warp(Template)
```

```text
bias = median(D)
```

```text
MAD = median(|D - bias|)
```

```text
sigmaDiff =
1.4826 × MAD / sqrt(2)
```

最终 Detection Threshold：

```text
T = K × sigmaDiff
```

如果启用 Local Noise：

```text
T(x,y) =
K × max(
    sigmaDiff,
    sigmaLocal(x,y)
)
```

最终：

```text
Defect(x,y) =
abs(D(x,y) - bias) > T(x,y)
```

---

# 101. Definition of Done

Codex/Cursor 完成后必须满足：

```text
[ ] CPU Global STD
[ ] CPU Global MAD
[ ] CPU Local STD
[ ] CPU Local MAD
[ ] CPU Difference STD
[ ] CPU Difference MAD

[ ] CUDA Histogram
[ ] CUDA Median
[ ] CUDA MAD
[ ] CUDA Difference
[ ] CUDA Local Noise
[ ] CUDA Reduction

[ ] Mask
[ ] Saturation handling
[ ] Bias correction
[ ] Sigma clamp
[ ] Auto fallback
[ ] Auto threshold

[ ] CPU/GPU numerical validation
[ ] Unit tests
[ ] Performance benchmark
[ ] Memory benchmark
[ ] No memory leak
[ ] CUDA error checking
[ ] Existing D2D regression test
```

---

# 102. Codex Execution Prompt

将以下内容直接作为 Cursor/Codex 执行指令：

```text
你现在是 Industrial Vision / CUDA Performance Engineer。

目标：

在当前工程中实现 NoiseEstimation 模块。

严格遵守：

1. 先扫描整个工程。
2. 找到现有：
   - Image abstraction
   - ROI
   - Mask
   - Histogram
   - Difference
   - Threshold
   - CUDA utility
   - CUDA stream
   - Memory pool
3. 优先复用已有基础设施。
4. 不允许重复实现已有基础模块。
5. 不修改现有 public API。
6. 如需要扩展 API，使用 overload 或新增接口。
7. CPU implementation 必须作为 reference。
8. CUDA implementation 必须与 CPU 结果进行数值验证。
9. 所有 CUDA implementation 放入 .cu。
10. Header 只保留必要 API / struct / enum。
11. 使用 C++17。
12. 优先使用 CUDA Runtime + CUB。
13. 禁止引入大型第三方依赖。

必须实现：

Global STD
Global MAD
Local STD
Local MAD
Difference STD
Difference MAD
Auto Noise Estimation
Bias Correction
Mask
Saturation Handling
Sigma Clamp
Adaptive Threshold
Fallback

D2D 默认算法：

D = Data - Warp(Template)

bias = median(D)

D' = D - bias

MAD = median(abs(D'))

sigmaDifference = 1.4826 * MAD / sqrt(2)

threshold = K * sigmaDifference

defect = abs(D') > threshold

Local Noise 可选：

sigmaLocal(x,y)

threshold(x,y) =
K * max(sigmaDifference, sigmaLocal(x,y))

CUDA：

使用 shared-memory / warp-level histogram。
避免大量 global atomic。
使用 CUB reduction。
使用 persistent workspace。
避免每个 ROI malloc/free。
避免不必要的 cudaDeviceSynchronize。
尽量减少 GPU/CPU data transfer。

必须添加：

1. Unit Test
2. CPU/CUDA consistency test
3. Synthetic Gaussian Noise test
4. Outlier robustness test
5. Difference Noise test
6. Mask test
7. Saturation test
8. Performance benchmark

完成后输出：

1. 修改文件列表
2. 新增文件列表
3. API 说明
4. 算法说明
5. CPU/CUDA 差异
6. Benchmark
7. 测试结果
8. 潜在性能瓶颈
9. 下一步优化建议

不要只给出示例代码。

必须直接修改当前工程代码并完成编译和测试。
```

---

# 103. Final Architecture

最终整个检测系统建议形成：

```text
                 ┌─────────────────────┐
                 │      Input ROI       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Intensity Normalize │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Alignment      │
                 │ Phase/ECC/Subpixel  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Warp Template    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Difference Image   │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
     ┌─────────────────┐         ┌─────────────────┐
     │ Difference MAD  │         │   Local Noise   │
     │ Bias + Robust σ │         │    MAD Map      │
     └────────┬────────┘         └────────┬────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 ┌─────────────────────┐
                 │   Noise Fusion       │
                 │    σ Effective       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Adaptive Threshold │
                 │      K × σ(x,y)     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Defect Binary     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Morphology / Blob   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Defect Classification│
                 └─────────────────────┘
```

**核心原则：**

```text
Noise Estimation
        ↓
不是一个独立的统计函数
        ↓
而是整个 D2D / SDI Detection Threshold 的统计基础
```

因此后续如果要继续往 **KLA 风格的实际工程链路**靠近，建议把它和你前面的 `IntensityNormalize`、`HistogramThreshold`、`Phase Correlation/ECC Alignment`、`Difference` 统一成一个 **Detection Statistics Layer**，而不是各模块各自计算一套 threshold/noise。