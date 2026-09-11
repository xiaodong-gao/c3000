# HistogramThreshold 逻辑结构

本文对应现有实现，不讨论改法。
职责：给模板/数据灰度图算一个阈值 T，再二值化，供相位对位使用。
ECC 对位走灰度图，不使用这里的二值图。

---

## 1. 代码结构（文件 → 职责）

```
utils/
  HistogramThreshold.h      公共类型、参数、结果、CPU C API、HistogramThresholdCpu
  HistogramThreshold.cpp    峰值/阈值规则（host 共用）+ CPU 实现
  HistogramThreshold.cuh    CUDA C API + 类 HistogramThreshold（CPU/CUDA 都包一层）
  HistogramThreshold.cu     GPU kernel、workspace、H2D/D2H；算完 hist 后回调 .cpp 里的规则

调用方
  app/die_types.h           Options.histogram / HistogramMethod
  app/die_inspect.cpp       解析 Cpu/Cuda/Auto，把 HistogramThreshold 放进 PipelineContext
  die2die/Pipeline.cpp      生产路径：computeFromGray + apply，再 fuse(T_tmpl, T_data)
  main.cpp                  配方：sigma / peakRatio / minDist / minCount / defaultT
```

两套入口、一套规则：

| 入口 | 文件 | 像素级工作 | 峰值与 T |
|------|------|------------|----------|
| `HistogramThresholdCpu` | `.h` / `.cpp` | CPU | `histogramThresholdComputeFromHistogram` |
| `HistogramThreshold` | `.cuh` / `.cu` | CUDA 或转调 CPU | 同上（host） |

生产只用 `HistogramThreshold`：CPU 调 `*Cpu` 方法，CUDA 调无后缀方法。

---

## 2. 在 D2D 管线中的位置

```
preparePair()                         OpenCV 转灰度（不是本模块 rgbToGray）
        │
        ▼
computeFromGray(template / data)      256-bin 计数 → 平滑 → 找峰 → 过滤 → T
        │
        ▼
apply(gray, T) → binary               gray > T ? 255 : 0
        │
        ▼
fuse(T_template, T_data)              (T_t + T_d) / 2   （整数）
        │
        ├─ Phase 对位  ← 用 templateBin / dataBin
        └─ ECC 对位    ← 仍用灰度，不用二值
```

后端选择（`die_inspect.cpp`）：

```
histogram_method
  Cpu   → PipelineBackend::Cpu
  Cuda  → PipelineBackend::Cuda
  Auto  → 跟随 align 后端
```

`Pipeline.cpp` 用 `ctx.overrideHistogramBackend` 固定这次选的后端，与对位后端可不同。

---

## 3. 数据类型

```
HistogramThresholdParams
  gaussianSigma        平滑 sigma，默认 2.5
  peakHeightRatio      相对最高峰的高度门限，默认 0.05
  minPeakDistance      保留峰之间最小灰度距离，默认 15
  minPeakCountRatio    峰高相对像素数下限，默认 0.005
  defaultThreshold     0 个有效峰时的 T，默认 125

HistogramPeak
  gray                 峰位置 0..255
  height               平滑后直方图高度

HistogramThresholdResult
  threshold            输出 T
  peakCount            过滤后峰数
  rawPeakCount         平滑后原始峰数
  rawPeaks[]           未过滤
  filteredPeaks[]      已过滤，按 gray 升序
  rule[]               用了哪条 T 规则
  smoothed[256]        平滑直方图
  histogram[256]       原始计数
```

常量：`HISTOGRAM_THRESHOLD_BINS = 256`，`HISTOGRAM_THRESHOLD_MAX_PEAKS = 64`。

---

## 4. 模块内部逻辑（CPU / CUDA 共用后半段）

```
输入图像（BGR 或 Gray，连续、n = w*h）
        │
        ├─ BGR 路径 ── rgbToGray ──► Gray
        │                 │
        │                 ▼
        └─ Gray 路径 ── histogram 256 bin
                          │
                          ▼
            histogramThresholdComputeFromHistogram   ← 只在 host
                          │
                          ├─ 转 float
                          ├─ 高斯平滑（镜像边界，半径 min(31, ceil(3σ))）
                          ├─ findPeaks
                          ├─ filterPeaks
                          └─ 按峰数选 T
                          │
                          ▼
                     Result.threshold
                          │
                          ▼
                     apply → 二值图
```

生产只走 Gray 路径：`computeFromGray` + `apply`。
BGR 路径（`rgbToGray` / `computeFromBgr`）是模块自带接口，管线不用。

---

## 5. 峰值与阈值规则（`.cpp`，CPU/CUDA 相同）

### 5.1 找峰 `findPeaks`（在平滑后的 256 点上）

```
端点 0：     hist[0] > hist[1]           → 峰
端点 255：   hist[255] > hist[254]       → 峰
中间：       上升平台后下降
             hist[i] > hist[i-1]
             再跳过一段相等
             hist[j] > hist[j+1]
             峰位 = (i+j)/2
```

### 5.2 过滤 `filterPeaks`

```
maxHeight = 原始峰里最高
heightGate = maxHeight * peakHeightRatio
countGate  = pixelCount * minPeakCountRatio

保留：height >= heightGate 且 height >= countGate
按 height 降序
贪心：与已选峰 |Δgray| < minPeakDistance 则丢
最后按 gray 升序
```

### 5.3 选 T

| 过滤后峰数 | T | rule |
|------------|---|------|
| 0 | `defaultThreshold`（默认 125） | `0 peaks -> T=125` |
| 1 | `(peak + 255) / 2` | `1 peak -> T=(peak+255)/2` |
| 2 | `(peak1 + peak2) / 2` | `2 peaks -> T=(peak1+peak2)/2` |
| >2 | 高度最高的两个峰的 gray 平均 | `>2 peaks -> T=(top1+top2)/2` |

除法都是整数。`fuse`：`(templateT + dataT) / 2`。

二值化：`gray[i] > T ? 255 : 0`（等于 T 为暗）。

---

## 6. CPU 结构

```
HistogramThresholdCpu
    rgbToGray            (77R + 150G + 29B) >> 8
    computeHistogram     ++hist[gray]
    computeFromGray      hist → ComputeFromHistogram
    computeFromBgr       rgbToGray → computeFromGray
    apply                gray > T
    fuse                 静态，整数平均
```

C 函数与类方法一一对应，失败返回 -1 / false。
要求 host 缓冲连续：`nPixels = width * height`，BGR 为 packed 3 通道。

---

## 7. CUDA 结构

```
HistogramThreshold                 持有 DeviceWorkspace*
    CUDA：rgbToGray / computeHistogram / computeFromGray / computeFromBgr / apply
    CPU： *Cpu 方法，转调 HistogramThreshold.cpp
    析构释放 workspace

C API（无对象时用进程内 defaultWorkspace）
    histogramThresholdRgbToGray
    histogramThresholdComputeHistogram
    histogramThresholdComputeFromGray
    histogramThresholdComputeFromBgr
    histogramThresholdApply
    histogramThresholdReleaseWorkspace
```

### 7.1 DeviceWorkspace

复用缓冲，按容量增长：`dBgr` / `dGray` / `dBin` / `dHist[256]`。

每次调用：H2D → kernel → D2H。生产接口是 host 指针，耗时含拷贝。

### 7.2 Kernel

| kernel | 作用 | 公式/细节 |
|--------|------|-----------|
| `rgbToGrayKernel` | BGR→灰 | `(77*p.z + 150*p.y + 29*p.x) >> 8`，uchar3，z=R |
| `histogramKernel` | 灰度计数 | block 内 8 份 shared hist + atomic，再归约到全局 256 bin |
| `bgrToGrayHistogramKernel` | 转灰并计数 | 上面两步合成，避免多一次遍历 |
| `applyKernel` | 二值化 | 先按 uchar4 向量化，尾部不足 4 像素标量处理 |

配置：`kBlockSize=256`，`kMaxGridSize=2048`，`kHistCopies=8`。
grid 按 `pixelsPerThread` 缩小：histogram=16，bgr+hist=4，apply=4。

CUDA 只替换「转灰 / 计数 / 二值」。平滑、找峰、选 T 仍在 host，与 CPU 同一函数，因此 bin 一致则 T 与二值一致。

---

## 8. API 对照

| 逻辑步骤 | CPU | CUDA（类方法） |
|----------|-----|----------------|
| BGR→灰 | `rgbToGrayCpu` | `rgbToGray` |
| 只计数 | `HistogramThresholdCpu::computeHistogram` | `computeHistogram` |
| 灰度→T | `computeFromGrayCpu` | `computeFromGray` |
| BGR→T | `computeFromBgrCpu` | `computeFromBgr` |
| 二值化 | `applyCpu` | `apply` |
| 融合 T | `HistogramThreshold::fuse` | 同左 |

管线实际调用：

```
CPU : computeFromGrayCpu(template) + computeFromGrayCpu(data)
      applyCpu(template, T_t) + applyCpu(data, T_d)
      fuse(T_t, T_d)

CUDA: computeFromGray(template) + computeFromGray(data)
      apply(template, T_t) + apply(data, T_d)
      fuse(T_t, T_d)
```

---

## 9. 配方（`main.cpp` → `opt.histogram`）

```
gaussianSigma      = 2.5
peakHeightRatio    = 0.05
minPeakDistance    = 15
minPeakCountRatio  = 0.005
defaultThreshold   = 125
histogram_method   = Cpu     （可改 Cuda / Auto）
```
