# KLA-style Die-to-Die Detection
## CUDA C++ Engineering Specification for Cursor / Codex

> 本文档用于指导 Cursor / Codex 对现有 C++ / CUDA 图像检测工程进行代码分析、修改和实现。
>
> **核心目标：**
> 在现有工程基础上实现稳定、高精度、可配置、GPU 加速的 Die-to-Die 缺陷检测流程。
>
> **重要：**
> - 不要直接重构整个工程。
> - 修改前必须先扫描现有代码结构。
> - 优先复用现有图像、CUDA、ROI、参数和结果结构。
> - 采用最小侵入式修改。
> - 不允许为了实现功能而破坏现有 API。
> - 不允许删除已有功能。
> - 如果现有模块已经存在，应优先扩展而不是重新创建重复模块。

---

# 1. Overall Objective

实现如下 Die-to-Die 检测流程：

```text
Template ROI
        │
        ▼
CV_8U Conversion + Gray
        │
        ▼
Histogram + Binary
        │
        ▼
Sub-pixel Alignment
Phase Correlation / ECC
        │
        ▼
Warp Data Original Image
        │
        ▼
Common Valid ROI
        │
        ▼
Per-channel Intensity Normalize
        │
        ▼
Template F / Data F
        │
        ▼
Difference
 ┌──────┼────────┐
 │      │        │
Abs   Signed     L2
 │      │        │
 │   ┌──┴──┐     │
 │   ▼     ▼     │
 │ Bright Dark   │
 │   │     │     │
 └───┴─────┴─────┘
        │
        ▼
Adaptive Threshold
        │
        ▼
Morphology
        │
        ▼
Connected Components
        │
        ▼
Defect Feature Extraction
        │
        ▼
Defect Filtering
        │
        ▼
Final Defect List
```

---

# 2. Core Design Principle

必须严格区分：

```text
辅助对位数据
```

和：

```text
最终缺陷检测数据
```

Histogram / Binary 主要用于：

```text
Alignment
```

不能使用 Binary 图替代原始灰度图进行最终 Difference。

正确的数据流：

```text
Original ROI
     │
     ├──────────────► Preserve Original
     │
     ▼
CV_8U
     │
     ▼
Gray
     │
     ▼
Histogram
     │
     ▼
Binary
     │
     ▼
Alignment
     │
     ▼
dx / dy
     │
     ▼
Warp Original Data
     │
     ▼
Common ROI
     │
     ▼
Intensity Normalize
     │
     ▼
Difference
     │
     ▼
Defect
```

---

# 3. Input

输入两张 ROI：

```cpp
cv::Mat templateROI;
cv::Mat dataROI;
```

支持：

```text
CV_8UC1
CV_8UC3
CV_16UC1
CV_16UC3
CV_32FC1
CV_32FC3
```

要求：

1. Template 和 Data 尺寸必须一致，或者在进入 Alignment 前进行有效尺寸检查。
2. 支持单通道和多通道。
3. 必须保留原始图像。
4. 不允许直接覆盖用户输入 Mat。
5. 尽量使用 `const cv::Mat&`。
6. 避免不必要的数据复制。

---

# 4. Convert to CV_8U

统一生成：

```cpp
cv::Mat template8u;
cv::Mat data8u;
```

对于：

```text
CV_8U
```

直接使用。

对于：

```text
CV_16U
CV_32F
```

根据实际动态范围进行缩放。

不要简单粗暴地：

```cpp
src.convertTo(dst, CV_8U);
```

导致高位图像信息截断。

建议支持：

```text
Direct
MinMax
Percentile
Robust
```

默认优先：

```text
Robust / Percentile
```

---

# 5. Gray Image Generation

分别生成：

```cpp
cv::Mat templateGray;
cv::Mat dataGray;
```

如果输入：

```text
1 channel
```

直接使用。

如果输入：

```text
3 channel
```

转换：

```cpp
cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
```

如果输入：

```text
4 channel
```

使用：

```cpp
cv::COLOR_BGRA2GRAY
```

Gray 图主要用于：

```text
Histogram
Binary
Alignment
```

不能代替原始多通道图进行最终检测。

---

# 6. Histogram Threshold

Template 和 Data 分别计算灰度直方图：

```text
Template Gray
      │
      ▼
Histogram
      │
      ▼
Threshold
      │
      ▼
Template Binary

Data Gray
      │
      ▼
Histogram
      │
      ▼
Threshold
      │
      ▼
Data Binary
```

必须支持独立 Threshold：

```cpp
float templateThreshold;
float dataThreshold;
```

不要强制两张图使用相同阈值。

---

# 7. Histogram Threshold Strategy

基础策略：

```text
没有明显波峰
    ↓
Threshold = 125

存在一个主要波峰
    ↓
根据 Peak → 255 区间确定 Threshold

存在多个波峰
    ↓
选择稳定主峰 / 自适应策略
```

要求将 Histogram Threshold 独立封装：

```cpp
class HistogramThreshold
{
public:
    float compute(const cv::Mat& gray);
    void computeHistogram(...);
};
```

不要把 Histogram 逻辑直接写进 DieToDieDetector。

---

# 8. Binary Generation

生成：

```cpp
cv::Mat templateBinary;
cv::Mat dataBinary;
```

例如：

```cpp
cv::threshold(
    templateGray,
    templateBinary,
    templateThreshold,
    255,
    cv::THRESH_BINARY
);
```

Binary 的用途：

```text
Alignment辅助
```

而不是：

```text
最终缺陷检测
```

---

# 9. Sub-pixel Alignment

支持两种方法：

```cpp
enum class AlignmentMethod
{
    PhaseCorrelation,
    ECC
};
```

---

## 9.1 Phase Correlation

默认优先用于：

```text
Translation
Sub-pixel Translation
```

输出：

```cpp
struct AlignmentResult
{
    double dx;
    double dy;
    double response;
    bool success;
};
```

要求：

```text
dx / dy
```

支持亚像素精度。

例如：

```text
dx = 0.37 px
dy = -0.21 px
```

---

# 10. ECC Alignment

ECC 用于更复杂的对位场景。

至少支持：

```text
Translation
Euclidean
Affine
```

建议默认：

```text
Translation
```

只有在现有工程明确存在旋转 / 仿射误差时才启用：

```text
Euclidean
Affine
```

避免不必要增加自由度导致误配。

---

# 11. Alignment Validation

Alignment 后必须进行质量检查。

至少检查：

```text
response
correlation
convergence
dx
dy
```

建议：

```cpp
bool validAlignment(
    const AlignmentResult& result,
    const AlignmentConfig& config
);
```

如果 Alignment 失败：

```text
禁止继续产生可信 Defect Result
```

必须返回：

```text
AlignmentFailed
```

或者由配置决定：

```text
FallbackToIntegerAlignment
```

---

# 12. Warp Data Original Image

获得：

```text
dx
dy
```

后：

```text
Data Original
      │
      ▼
WarpAffine
      │
      ▼
Aligned Data
```

重要：

**必须 Warp 原始 Data ROI，而不是 Binary Data。**

即：

```text
Data Original
      ↓
Warp
      ↓
Normalize
      ↓
Difference
```

而不是：

```text
Data Binary
      ↓
Warp
      ↓
Difference
```

---

# 13. Common Valid ROI

Warp 后边缘存在：

```text
invalid pixels
```

不能参与 Difference。

必须计算：

```cpp
cv::Rect commonROI;
```

或者：

```cpp
struct CommonROI
{
    int x;
    int y;
    int width;
    int height;
};
```

要求：

```text
Template Common ROI
=
Data Common ROI
```

尺寸完全一致。

---

# 14. Common ROI Calculation

流程：

```text
Template ROI
      │
      ├──────────────┐
      │              │
      ▼              │
Data ROI             │
      │              │
      ▼              │
Warp                 │
      │              │
      ▼              │
Valid Mask            │
      │              │
      └──────┬───────┘
             ▼
       Common Valid ROI
```

禁止使用 Warp 后产生的：

```text
0
255
```

等填充值直接参与 Difference。

---

# 15. Per-channel Intensity Normalize

Common ROI 后进行：

```text
逐通道亮度归一化
```

对于：

```text
Gray
```

处理：

```text
Template Gray
Data Gray
```

对于：

```text
BGR
```

分别：

```text
B
G
R
```

---

# 16. Normalization Methods

建议支持：

```cpp
enum class NormalizeMethod
{
    Mean,
    MeanStd,
    MinMax,
    Percentile,
    LinearRegression,
    RobustLinear
};
```

推荐默认：

```text
RobustLinear
```

或者：

```text
Percentile
```

避免少量 Defect 对统计参数产生明显影响。

---

# 17. Normalization Formula

基础 Mean：

```text
scale = templateMean / dataMean

dataNormalized = data * scale
```

Mean + Std：

```text
dataNormalized =
    (data - dataMean)
    * templateStd / dataStd
    + templateMean
```

Robust Percentile：

```text
P01
P50
P99
```

基于：

```text
Template
Data
```

计算 robust mapping。

---

# 18. Important Normalization Rule

归一化统计参数不能被明显 Defect 大量污染。

推荐：

```text
统计参数
    ↓
Exclude extreme pixels
    ↓
Percentile / Robust statistics
```

例如：

```text
P01
P50
P99
```

不要直接依赖：

```text
min
max
```

因为一个异常 Pixel 就可能严重改变 Normalize 参数。

---

# 19. Difference Engine

必须支持：

```cpp
enum class DifferenceMethod
{
    Absolute,
    Signed,
    L2
};
```

---

# 20. Absolute Difference

公式：

```text
Dabs = |Data - Template|
```

OpenCV：

```cpp
cv::absdiff(data, templ, diff);
```

用于：

```text
快速候选缺陷
```

---

# 21. Signed Difference

核心公式：

```text
D = Data - Template
```

必须保留正负号。

不要：

```cpp
abs(Data - Template)
```

覆盖 Signed Difference。

---

# 22. Bright / Dark Classification

定义：

```text
Bright:
    D > +Tbright

Dark:
    D < -Tdark
```

代码逻辑：

```cpp
brightMask = signedDiff > brightThreshold;
darkMask   = signedDiff < -darkThreshold;
```

---

# 23. Bright Defect

Bright Defect 定义：

```text
Data > Template
```

即：

```text
signedDiff > +Tbright
```

输出：

```cpp
DefectType::Bright
```

---

# 24. Dark Defect

Dark Defect 定义：

```text
Data < Template
```

即：

```text
signedDiff < -Tdark
```

输出：

```cpp
DefectType::Dark
```

---

# 25. L2 Difference

对于多通道：

```text
D =
sqrt(
    dB² +
    dG² +
    dR²
)
```

即：

```cpp
float db = dataB - templateB;
float dg = dataG - templateG;
float dr = dataR - templateR;

float l2 = sqrtf(
    db * db +
    dg * dg +
    dr * dr
);
```

L2 用于：

```text
多通道综合差异
```

---

# 26. Threshold

Threshold 至少支持：

```text
Fixed
Adaptive
Noise-based
Local
Global
```

建议：

```text
Tbright = f(localNoise, globalNoise, background)
Tdark   = f(localNoise, globalNoise, background)
```

即：

```text
Bright Threshold
和
Dark Threshold
```

必须可以独立配置。

---

# 27. Threshold Architecture

建议：

```cpp
struct ThresholdConfig
{
    float brightThreshold;
    float darkThreshold;

    bool adaptive;
    float noiseFactor;

    int localWindowSize;
};
```

不要把 Threshold 常数硬编码到 Kernel。

---

# 28. Morphology

Bright / Dark 分开处理：

```text
Bright Mask
    ↓
Morphology
    ↓
Bright Clean Mask

Dark Mask
    ↓
Morphology
    ↓
Dark Clean Mask
```

支持：

```text
Open
Close
Dilate
Erode
Remove Small Components
```

参数：

```cpp
int kernelSize;
int iterations;
```

必须可配置。

---

# 29. Connected Components

对：

```text
Bright Clean Mask
Dark Clean Mask
```

分别执行：

```text
Connected Components
```

生成：

```cpp
DefectBlob
```

---

# 30. DefectBlob

建议结构：

```cpp
enum class DefectType
{
    Bright,
    Dark
};

struct DefectBlob
{
    int x;
    int y;
    int width;
    int height;

    float area;
    float perimeter;

    float meanDiff;
    float maxDiff;
    float score;

    float aspectRatio;

    DefectType type;
};
```

---

# 31. Defect Features

至少计算：

```text
Bounding Box
Area
Width
Height
Aspect Ratio
Mean Difference
Max Difference
Score
Centroid
```

可扩展：

```text
Perimeter
Circularity
Local Contrast
Density
Edge Distance
```

---

# 32. Defect Filtering

支持：

```text
Min Area
Max Area
Min Width
Max Width
Min Height
Max Height
Min Score
Max Aspect Ratio
Min Aspect Ratio
Edge Filter
```

示例：

```cpp
if (blob.area < config.minArea)
    reject();

if (blob.area > config.maxArea)
    reject();
```

---

# 33. Final Result

最终结果必须同时保留：

```text
Bright Defects
Dark Defects
```

建议：

```cpp
struct DieToDieResult
{
    bool success;

    AlignmentResult alignment;

    cv::Rect commonROI;

    std::vector<DefectBlob> brightDefects;
    std::vector<DefectBlob> darkDefects;

    int totalDefects;

    float processingTimeMs;
};
```

---

# 34. Main API

最终对外接口尽量保持简单：

```cpp
class DieToDieDetector
{
public:

    DieToDieResult detect(
        const cv::Mat& templateROI,
        const cv::Mat& dataROI
    );
};
```

配置：

```cpp
void setConfig(
    const DieToDieConfig& config
);
```

或者：

```cpp
DieToDieResult detect(
    const cv::Mat& templateROI,
    const cv::Mat& dataROI,
    const DieToDieConfig& config
);
```

优先复用现有工程 API。

---

# 35. Configuration

建议：

```cpp
struct DieToDieConfig
{
    // Input
    bool autoConvert8U = true;

    // Histogram
    bool enableHistogram = true;
    bool enableBinary = true;

    // Alignment
    AlignmentMethod alignmentMethod =
        AlignmentMethod::PhaseCorrelation;

    bool enableSubpixel = true;

    double alignmentEpsilon = 1e-6;
    int alignmentIterations = 100;

    // Normalize
    NormalizeMethod normalizeMethod =
        NormalizeMethod::RobustLinear;

    bool perChannelNormalize = true;

    // Difference
    bool enableAbsoluteDifference = true;
    bool enableSignedDifference = true;
    bool enableL2Difference = true;

    // Threshold
    bool adaptiveThreshold = true;
    float brightThreshold = 10.0f;
    float darkThreshold = 10.0f;

    // Morphology
    bool enableMorphology = true;
    int morphologyKernelSize = 3;
    int morphologyIterations = 1;

    // Blob
    float minArea = 3.0f;
    float maxArea = 100000.0f;

    // Filter
    bool enableEdgeFilter = true;
};
```

具体默认值必须根据现有工程和实际数据验证，不能盲目覆盖现有参数。

---

# 36. CUDA Architecture

CUDA 实现必须遵循：

```text
CPU
 │
 ├── Input Validation
 ├── Configuration
 ├── Alignment Control
 └── Result Management
        │
        ▼
GPU
 │
 ├── CV_8U Conversion
 ├── Gray
 ├── Histogram
 ├── Binary
 ├── Warp
 ├── Normalize
 ├── Signed Difference
 ├── Absolute Difference
 ├── L2
 ├── Threshold
 ├── Morphology
 └── Feature Calculation
        │
        ▼
CPU
 │
 └── Final Defect Result
```

---

# 37. CUDA Memory Rules

重点优化：

```text
减少 Host ↔ Device Copy
减少 GPU Memory Allocation
避免每个阶段 cudaMalloc
避免每个阶段 cudaFree
```

建议：

```cpp
class DieToDieCudaContext
{
public:
    void reserve(
        int width,
        int height,
        int channels
    );

    void release();
};
```

Buffer 复用：

```text
d_template
d_data
d_grayTemplate
d_grayData
d_binaryTemplate
d_binaryData
d_alignedData
d_diff
d_signedDiff
d_brightMask
d_darkMask
```

---

# 38. CUDA Stream

支持：

```cpp
cudaStream_t stream;
```

如果没有数据依赖，可以考虑：

```text
Template preprocessing
        │
        ├──── GPU Stream 0
        │
Data preprocessing
        │
        └──── GPU Stream 1
```

但是：

**不要为了并行而并行。**

必须根据实际：

```text
GPU occupancy
Memory bandwidth
Kernel dependency
```

决定。

---

# 39. CUDA Kernel Design

建议拆分：

```text
convertTo8UKernel
grayKernel
histogramKernel
thresholdKernel
warpKernel
normalizeKernel
signedDifferenceKernel
absoluteDifferenceKernel
l2DifferenceKernel
brightThresholdKernel
darkThresholdKernel
morphologyKernel
```

不要创建大量极小 Kernel 导致：

```text
Kernel Launch Overhead
```

对于连续操作可以考虑 Fusion。

---

# 40. Kernel Fusion

性能优化优先考虑：

```text
Normalize
    +
Difference
    +
Threshold
```

融合：

```text
normalizeDifferenceThresholdKernel
```

例如：

```text
Data
Template
   │
   ▼
Normalize
   │
   ▼
Signed Difference
   │
   ▼
Bright / Dark Threshold
```

可以在一个 Kernel 内完成。

但是：

**只有 Benchmark 证明有收益时才进行 Fusion。**

---

# 41. CPU / GPU Responsibility

CPU：

```text
Config
Validation
ROI
Algorithm Selection
Alignment Control
Result
Blob Management
```

GPU：

```text
Pixel-level operations
Histogram
Normalize
Difference
Threshold
Morphology
```

---

# 42. Performance Requirements

修改代码时必须关注：

```text
CPU usage
GPU usage
Memory bandwidth
GPU memory
Host memory
Allocation count
Kernel launch count
PCIe transfer
Latency
Throughput
```

必须尽量避免：

```cpp
cv::Mat clone()
```

大量出现。

避免：

```cpp
cudaMalloc
cudaFree
```

出现在每次 Detect 调用内部。

---

# 43. Zero / Low Copy Principle

理想流程：

```text
Input
 ↓
Upload GPU
 ↓
GPU Processing
 ↓
Download Final Result
```

而不是：

```text
Input
 ↓
GPU
 ↓
CPU
 ↓
GPU
 ↓
CPU
 ↓
GPU
 ↓
CPU
```

尽量减少：

```text
Host ↔ Device
```

数据传输。

---

# 44. Error Handling

必须处理：

```text
Empty ROI
Invalid ROI
Size mismatch
Unsupported type
Unsupported channel
Alignment failure
CUDA allocation failure
CUDA kernel failure
Invalid configuration
No valid common ROI
```

禁止：

```cpp
assert()
```

作为生产环境唯一错误处理。

---

# 45. Logging

建议提供：

```text
Alignment dx
Alignment dy
Alignment response

Template threshold
Data threshold

Common ROI

Normalization parameters

Bright threshold
Dark threshold

Bright defect count
Dark defect count

GPU processing time
CPU processing time
Total processing time
```

Debug 模式可以输出：

```text
Template Gray
Data Gray
Binary
Aligned Data
Common ROI
Normalized Template
Normalized Data
Signed Difference
Bright Mask
Dark Mask
Final Blob
```

Release 模式禁止产生大量日志。

---

# 46. Debug Output

建议增加：

```cpp
struct DieToDieDebugResult
{
    cv::Mat templateGray;
    cv::Mat dataGray;

    cv::Mat templateBinary;
    cv::Mat dataBinary;

    cv::Mat alignedData;

    cv::Mat signedDifference;

    cv::Mat brightMask;
    cv::Mat darkMask;
};
```

必须通过：

```cpp
config.enableDebugOutput
```

控制。

---

# 47. Processing Pipeline

最终实现必须保持以下顺序：

```text
[01] Input ROI
      ↓
[02] Convert CV_8U
      ↓
[03] Generate Gray
      ↓
[04] Histogram
      ↓
[05] Binary
      ↓
[06] Phase Correlation / ECC
      ↓
[07] Sub-pixel Alignment
      ↓
[08] Warp Data Original
      ↓
[09] Common Valid ROI
      ↓
[10] Per-channel Intensity Normalize
      ↓
[11] Absolute Difference
      ↓
[12] Signed Difference
      ↓
[13] L2 Difference
      ↓
[14] Bright / Dark Threshold
      ↓
[15] Morphology
      ↓
[16] Connected Components
      ↓
[17] Feature Extraction
      ↓
[18] Defect Filtering
      ↓
[19] Bright Defect
      +
[20] Dark Defect
      ↓
[21] Final Result
```

---

# 48. Important Algorithm Constraints

## Constraint 1

不能使用：

```cpp
abs(data - template)
```

作为唯一 Difference。

必须保留：

```cpp
signedDiff = data - template;
```

---

## Constraint 2

必须同时输出：

```text
Bright Defect
Dark Defect
```

---

## Constraint 3

Binary 主要用于：

```text
Alignment
```

不能替代原始灰度图。

---

## Constraint 4

Warp 必须作用于：

```text
Data Original ROI
```

而不是 Binary。

---

## Constraint 5

Difference 必须在：

```text
Common Valid ROI
```

上执行。

---

## Constraint 6

Normalize 必须在：

```text
Alignment
+
Common ROI
```

之后执行。

---

## Constraint 7

多通道图像必须支持：

```text
Per-channel Normalize
```

---

## Constraint 8

Bright / Dark Threshold 必须独立：

```text
Tbright
Tdark
```

---

# 49. Recommended Source Structure

如果现有工程没有对应模块，建议：

```text
DieToDie/
│
├── DieToDieDetector.h
├── DieToDieDetector.cpp
│
├── DieToDieConfig.h
├── DieToDieResult.h
│
├── Preprocess/
│   ├── ImageConvert.h
│   ├── ImageConvert.cpp
│   ├── ImageConvert.cuh
│   └── ImageConvert.cu
│
├── Histogram/
│   ├── HistogramThreshold.h
│   ├── HistogramThreshold.cpp
│   ├── HistogramThreshold.cuh
│   └── HistogramThreshold.cu
│
├── Alignment/
│   ├── Alignment.h
│   ├── Alignment.cpp
│   ├── Alignment.cuh
│   └── Alignment.cu
│
├── Normalize/
│   ├── IntensityNormalize.h
│   ├── IntensityNormalize.cpp
│   ├── IntensityNormalize.cuh
│   └── IntensityNormalize.cu
│
├── Difference/
│   ├── Difference.h
│   ├── Difference.cpp
│   ├── Difference.cuh
│   └── Difference.cu
│
├── Threshold/
│   ├── AdaptiveThreshold.h
│   ├── AdaptiveThreshold.cpp
│   ├── AdaptiveThreshold.cuh
│   └── AdaptiveThreshold.cu
│
├── Morphology/
│   ├── Morphology.h
│   ├── Morphology.cpp
│   ├── Morphology.cuh
│   └── Morphology.cu
│
├── Blob/
│   ├── ConnectedComponents.h
│   ├── ConnectedComponents.cpp
│   ├── ConnectedComponents.cuh
│   └── ConnectedComponents.cu
│
└── Filter/
    ├── DefectFilter.h
    └── DefectFilter.cpp
```

但是：

> 如果现有工程已经存在对应模块，必须优先复用现有模块，不得机械创建上述目录。

---

# 50. Cursor / Codex Execution Rules

AI 修改代码时严格执行：

## Step 1 - Scan

先扫描：

```text
项目目录
CMake
CUDA
OpenCV
已有 ROI
已有 Normalize
已有 Histogram
已有 Alignment
已有 Difference
已有 Blob
已有 Result
```

---

## Step 2 - Build Architecture Map

输出：

```text
现有模块
调用关系
数据流
CUDA 数据流
主要接口
可复用代码
需要新增代码
```

---

## Step 3 - Identify Existing Implementation

重点搜索：

```text
IntensityNormalize
AutoGrayThreshold
Histogram
PhaseCorrelation
ECC
WarpAffine
ROI
Difference
ConnectedComponents
Morphology
Defect
CUDA
```

如果已经存在：

```text
AutoGrayThreshold
```

必须优先复用。

---

# 51. Modification Strategy

修改优先级：

```text
P0
功能正确性

P1
明暗缺陷正确区分

P2
Alignment 稳定性

P3
Normalize 稳定性

P4
误检率 / 漏检率

P5
CUDA 性能

P6
代码架构优化
```

不要一开始为了性能大规模重构。

---

# 52. Validation

至少准备以下测试：

```text
Test 01
Perfect Match

Test 02
Pure Translation

Test 03
Sub-pixel Translation

Test 04
Global Brightness Change

Test 05
Global Darkness Change

Test 06
Bright Defect

Test 07
Dark Defect

Test 08
Bright + Dark Defects

Test 09
Noise

Test 10
Low Contrast Defect

Test 11
Multi-channel

Test 12
Different Dynamic Range

Test 13
ROI Boundary

Test 14
Alignment Failure

Test 15
Large ROI
```

---

# 53. Acceptance Criteria

实现完成后必须验证：

```text
[ ] 两张 ROI 可以正确输入
[ ] 支持 CV_8U
[ ] 支持 Gray
[ ] Histogram 正确
[ ] Binary 正确
[ ] Phase Correlation 正确
[ ] ECC 正确
[ ] 支持亚像素平移
[ ] Warp Data 正确
[ ] Common ROI 正确
[ ] Per-channel Normalize 正确
[ ] Absolute Difference 正确
[ ] Signed Difference 正确
[ ] L2 Difference 正确
[ ] Bright Mask 正确
[ ] Dark Mask 正确
[ ] Morphology 正确
[ ] Connected Components 正确
[ ] Blob Feature 正确
[ ] Bright Defect 正确
[ ] Dark Defect 正确
[ ] CUDA 无明显 Memory Leak
[ ] CUDA Error Check 完整
[ ] CPU / GPU 结果一致
[ ] Debug Output 可关闭
[ ] Release 性能满足要求
```

---

# 54. Benchmark

必须记录：

```text
Input Width
Input Height
Channels

Preprocess Time
Histogram Time
Alignment Time
Warp Time
Normalize Time
Difference Time
Threshold Time
Morphology Time
Blob Time

CPU Total
GPU Total
End-to-End Total
```

输出：

```text
Before
After
```

并比较：

```text
Latency
Throughput
Memory
GPU Utilization
CPU Utilization
```

---

# 55. Final Engineering Requirement

最终实现必须满足：

```text
Template ROI
+
Data ROI
        ↓
统一 CV_8U
        ↓
Gray
        ↓
Histogram / Binary
        ↓
Phase Correlation / ECC
        ↓
Sub-pixel Alignment
        ↓
Warp Data Original
        ↓
Common Valid ROI
        ↓
Per-channel Intensity Normalize
        ↓
Absolute / Signed / L2 Difference
        ↓
Bright / Dark Separation
        ↓
Adaptive Threshold
        ↓
Morphology
        ↓
Connected Components
        ↓
Feature Extraction
        ↓
Defect Filtering
        ↓
Bright Defects + Dark Defects
```

核心要求：

```text
1. 不丢失 Signed Difference
2. 必须区分 Bright / Dark
3. Binary 仅辅助 Alignment
4. Warp 原始 Data
5. Common ROI 后再 Difference
6. Normalize 支持逐通道
7. Threshold 支持 Bright / Dark 独立参数
8. CUDA 尽量减少 Host ↔ Device Copy
9. GPU Buffer 尽量复用
10. 修改现有工程必须最小侵入
11. 优先复用已有模块
12. 修改后必须 Build + Test + Benchmark
```

---

# 56. Codex / Cursor Final Instruction

在开始修改代码之前，必须：

```text
1. 扫描整个项目。
2. 找到现有 Die-to-Die / ROI / Normalize / Histogram / CUDA 相关实现。
3. 分析现有架构。
4. 列出需要修改的文件。
5. 列出需要新增的文件。
6. 判断哪些模块可以直接复用。
7. 不允许重复实现已有功能。
8. 先实现正确性，再优化性能。
9. 修改后自动编译。
10. 自动运行已有测试。
11. 如果没有测试，则创建最小可执行测试。
12. 输出修改前后性能对比。
13. 输出所有修改文件和修改原因。
14. 不允许删除无关代码。
15. 不允许修改与本任务无关的功能。
```

最终报告必须包含：

```text
## Modified Files

## Added Files

## Architecture Changes

## Algorithm Changes

## CUDA Changes

## Test Results

## Benchmark Results

## Known Issues

## Next Optimization Suggestions
```

**End of Specification**