# PhaseCorrelate 逻辑结构

本文对应现有实现，不讨论改法。
职责：估计 data 相对 template 的平移 `(dx, dy)`（+x 右、+y 下）。
`shiftImage(data, shift)` 把 data 对齐到 template。约定与 EccAlign 相同。

生产路径喂的是直方图二值图，不是灰度图。

---

## 1. 代码结构（文件 → 职责）

```
ams/
  PhaseCorrelate.hpp    CPU 类、commonOverlapRoi、shiftImage
  PhaseCorrelate.cpp    cv::phaseCorrelate + 窗缓存；平移/重叠 ROI
  PhaseCorrelate.cuh    CUDA 结果、C API、类 PhaseCorrelateCuda
  PhaseCorrelate.cu     对齐 OpenCV phaseCorrelate：Hann、补零 DFT、互功率谱、fftShift、5×5 质心

调用方
  app/die_inspect.cpp       AlignMethod::PhaseCpu / PhaseCuda → PipelineContext
  die2die/Pipeline.cpp      二值图上算 shift，再 shiftImage / commonOverlapRoi
  main.cpp                  align_method = PhaseCpu（默认）或 PhaseCuda
```

两套入口、同一套相位相关步骤：

| 入口 | 实现 | 窗 |
|------|------|----|
| `PhaseCorrelate::compute` | OpenCV `cv::phaseCorrelate` | `createHanningWindow`，按原图尺寸缓存 |
| `PhaseCorrelateCuda::compute` | cuFFT D2Z / Z2D | 在补零后的 DFT 网格上乘 Hann |

`commonOverlapRoi` / `shiftImage` 放在本模块，Phase 和 ECC 对齐之后都用。

---

## 2. 在 D2D 管线中的位置

```
preparePair()                         转灰度
        │
        ▼
HistogramThreshold                    灰度 → 二值 templateBin / dataBin
        │
        ▼
align == Phase
        │
        ├─ PhaseCpu  → phase->compute(templateBin, dataBin, &response)
        └─ PhaseCuda → phaseCuda->compute(bin.data, bin.data, w, h, result)
        │
        ▼
out.shift = (dx, dy)
out.response = 5×5 窗内相关峰质量和（再除以 DFT 面积）
        │
        ▼
shiftImage(dataBin,  shift, INTER_NEAREST)     二值
shiftImage(dataOrig, shift, INTER_LINEAR)      原图
commonOverlapRoi(size, shift)                  有效重叠，宽高 < 16 则失败
        → 归一化 → 差分检出
```

`AlignMethod`：

```
PhaseCpu   → PipelineBackend::Cpu  + PipelineAlign::Phase
PhaseCuda  → PipelineBackend::Cuda + PipelineAlign::Phase
Ecc*       → 不走本模块，走 EccAlign（灰度）
```

管线里 `useHanning` 用默认 `true`。

---

## 3. 数据类型

CPU：`compute` 直接返回 `cv::Point2d`，可选写出 `response`。

CUDA：

```
PhaseCorrelateResult
  dx, dy        平移，与 CPU Point2d 同约定
  response      相关峰响应
```

位移约定（`PhaseCorrelate.hpp`）：

```
data 相对 template 的平移：+x 右、+y 下
shiftImage(data, shift) 把 data 对齐到 template
```

`shiftImage` 的仿射是：

```
[ 1  0  -dx ]
[ 0  1  -dy ]
warpAffine(..., BORDER_CONSTANT, 0)
```

即 `dst(x,y) = src(x-dx, y-dy)`（默认前向映射）。

`commonOverlapRoi(size, shift)`：

```
x0 = ceil(max(0, -dx))
y0 = ceil(max(0, -dy))
x1 = floor(min(w, w-dx))
y1 = floor(min(h, h-dy))
ROI = (x0, y0, x1-x0, y1-y0)    宽或高为负时收成 0
```

---

## 4. 模块内部逻辑（与 OpenCV `cv::phaseCorrelate` 对齐）

```
输入 template / data（单通道，同尺寸；生产为 U8 二值）
        │
        ▼
转 double（CPU：CV_64F；CUDA：U8 → double）
        │
        ▼
Hanning 窗（useHanning=true）
        │
        ▼
补零到 getOptimalDFTSize / optimalDftSize（2^p·3^q·5^r）
        │
        ▼
正向 DFT
        │
        ▼
互功率谱  C = (FA · conj(FB)) / |FA · conj(FB)|
        │
        ▼
逆向 DFT → 空间相关图
        │
        ▼
fftShift（零频移到中心，含奇数尺寸）
        │
        ▼
全局峰值 + 5×5（半径 2）加权质心
        │
        ▼
dx = dftW/2 - subX
dy = dftH/2 - subY
response = 窗内和 / (dftW * dftH)
```

CPU 窗乘在**原图**上，再由 OpenCV 内部补零做 DFT。
CUDA 窗乘在**补零后的 DFT 网格**上：原图外为 0，Hann 的分母是 `dftW-1` / `dftH-1`。

---

## 5. CPU 结构（`PhaseCorrelate`）

```
校验：非空、同尺寸、单通道

convertTo CV_64F

useHanning == false
    → cv::phaseCorrelate(templateF, dataF, noArray, response)

useHanning == true
    → 若 hann_ 尺寸/类型不是 CV_64F 原图大小，则 createHanningWindow
    → cv::phaseCorrelate(templateF, dataF, hann_, response)
```

`hann_` 为 `mutable`，同尺寸重复调用会复用窗。
`createHanningWindow`：各维 `0.5*(1-cos(2π i/(n-1)))`，二维相乘后再 `sqrt`。

---

## 6. CUDA 结构

```
PhaseCorrelateCuda
  持有 DeviceWorkspace*
  compute(..., useHanning=true) → computeDevice
  析构释放 workspace 与 cuFFT plan

C API（无对象时用 defaultWorkspace）
  phaseCorrelateCompute          固定 useHanning=true
  phaseCorrelateReleaseWorkspace
```

宽或高 `<= 1` → `cudaErrorInvalidValue`。

### 6.1 DeviceWorkspace

按原图 + 补零尺寸复用：

```
dTmplU8 / dDataU8     输入 U8（一块 nSrc*2）
dTmpl / dData / dCorr  double，各 nDft（一块 nDft*3）
dFft1 / dFft2         cufftDoubleComplex，各 paddedH*(paddedW/2+1)
planFwd               CUFFT_D2Z
planInv               CUFFT_Z2D
```

DFT 尺寸变化时销毁并重建 plan。

### 6.2 补零尺寸 `optimalDftSize`

与 `cv::getOptimalDFTSize` 相同：不小于 `n0` 的 `2^p * 3^q * 5^r`，搜索上限 `n0*2+32`，找不到则退回 `n0`。

### 6.3 Kernel / host 步骤

| 步骤 | 位置 | 作用 |
|------|------|------|
| H2D U8 | host | 两幅原图 |
| `padAndWindowKernel` | device | 写入 `dftW×dftH`：图内为像素，图外 0；可选 Hann |
| `cufftExecD2Z` | device | template / data 实数 FFT |
| `crossPowerKernel` | device | `FA * conj(FB)` 单位化 |
| `cufftExecZ2D` | device | 逆 FFT → `dCorr`（未除 N） |
| D2H + `fftShift2D` | host | 零频到中心 |
| 扫峰值 | host | 最大相关值 |
| `weightedCentroid` | host | 峰值周围 5×5 加权质心 |

Hann（`useHanning`）：

```
wx = 0.5 * (1 - cos(2π x / (dftW-1)))
wy = 0.5 * (1 - cos(2π y / (dftH-1)))
dst = val * sqrt(wx * wy)          与 OpenCV 先乘再 sqrt 一致
```

互功率谱：

```
re = a.x*b.x + a.y*b.y             Re(FA · conj(FB))
im = a.y*b.x - a.x*b.y             Im(FA · conj(FB))
mag = hypot(re, im) + 1e-15
C = (re, im) / mag
```

`a` = FFT(template)，`b` = FFT(data)。

`fftShift2D`：偶尺寸四块对调；奇尺寸按 OpenCV 把多出的一行/列放到前半块。

质心半径 `kCentroidRadius = 2`（5×5），`sum += 1e-15` 避免除零。
`response` 再除以 `dftW * dftH`。

---

## 7. API 对照

| 逻辑 | CPU | CUDA |
|------|-----|------|
| 计算平移 | `phase->compute(tmpl, data, &resp, useHanning)` | `phaseCuda->compute(tmpl, data, w, h, result, useHanning)` |
| 位移 | 返回 `Point2d` | `result.dx / result.dy` |
| 响应 | 输出参数 `response` | `result.response` |
| 默认窗 | `useHanning=true` | 同左；C API 固定 true |

管线实际调用：

```
CPU : ctx.phase->compute(out.templateBin, out.dataBin, &out.response)
CUDA: ctx.phaseCuda->compute(templateBin.data, dataBin.data, w, h, phase)
      out.shift = (phase.dx, phase.dy)
```

CUDA 要求 `.data` 连续，`n = w*h`。

---

## 8. 配方（`main.cpp`）

```
align_method = PhaseCpu     改 PhaseCuda 走 GPU
无独立 Phase 参数；窗默认开
直方图阈值决定二值图，从而影响本模块输入
```

---

## 9. 与 HistogramThreshold / EccAlign 的边界

| 模块 | 输入 | 输出用途 |
|------|------|----------|
| HistogramThreshold | 灰度 | 二值图，给 Phase |
| PhaseCorrelate | 二值 | shift，给 `shiftImage` |
| EccAlign | 灰度 | shift，给 `shiftImage` |

三者位移约定相同。Phase 与 Ecc 在管线里互斥，由 `PipelineAlign` 选择。
重叠 ROI 与 `shiftImage` 由本文件提供，两种对位共用。
