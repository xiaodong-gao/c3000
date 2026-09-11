# IntensityNormalize 逻辑结构

本文对应现有实现，不讨论改法。
职责：在**已对齐的重叠 ROI** 上，把 data 的辐射亮度匹配到 template，再算逐像素绝对差和 L2 差，供后续缺陷检出。

模板图本身不做辐射变换，只拷贝。data 按通道独立做 `gain * x + bias`。统计用中位数 + MAD 剔除大缺陷，避免增益被异常点拉偏。

---

## 1. 代码结构（文件 → 职责）

```
utils/
  IntensityNormalize.h      统计/输出结构、robustMeanStdFromU8Histogram、CPU/CUDA 封装函数
  IntensityNormalize.cpp    校验、CPU 直方图统计、convertTo 映射、absdiff / L2
  IntensityNormalize.cuh    CUDA C API、DeviceView、类 IntensityNormalizeCuda
  IntensityNormalize.cu     GPU 直方图 + 映射/差分；增益偏差仍用 host 共用规则

调用方
  app/die_inspect.cpp       NormalizeMethod Cpu/Cuda/Auto → PipelineContext.intensityCuda
  die2die/Pipeline.cpp      重叠 ROI 上调用 intensityNormalize 或 intensityNormalizeCuda
  main.cpp                  normalize_method（算法无独立配方参数）
```

两套入口、同一套稳健统计：

| 入口 | 像素级工作 | 均值/方差 / gain / bias |
|------|------------|-------------------------|
| `intensityNormalize` | CPU 直方图、`convertTo`、`absdiff` | `robustMeanStdFromU8Histogram` |
| `intensityNormalizeCuda` | GPU 直方图、映射、差分 | 同上（host） |

---

## 2. 在 D2D 管线中的位置

```
对位 + shiftImage(data)
        │
        ▼
commonOverlapRoi                    只在有效重叠上做辐射匹配
        │                           （避免 warp 黑边污染统计）
        ▼
intensityNormalize(templateCommon, dataCommon)
        │
        ├─ templateNormalized = 模板拷贝（不改亮度）
        ├─ dataNormalized     = data * gain + bias（每通道）
        ├─ absoluteDifference 逐通道 |t - d|
        └─ l2Difference       通道欧氏距离，CV_32FC1
        │
        ▼
极性 / 自适应阈值 / 形态学 / blob
```

后端选择（`die_inspect.cpp`）：

```
normalize_method
  Cpu   → PipelineBackend::Cpu
  Cuda  → PipelineBackend::Cuda
  Auto  → 跟随 align 后端
```

CUDA 路径需要 `ctx.intensityCuda`；`Pipeline.cpp` 用 `overrideNormalizeBackend` 固定本次后端，可与对位后端不同。

---

## 3. 数据类型

```
IntensityNormalizeStats          每通道最多 4 槽，实际用 1 或 3
  channels
  templateMean / templateStdDev
  dataMean / dataStdDev
  gain / bias

IntensityNormalizeOutput
  ok, error
  templateNormalized     与输入同类型，参考图未改
  dataNormalized         CV_8U，同通道数
  absoluteDifference     CV_8U，同通道数
  l2Difference           CV_32FC1
  stats

IntensityNormalizeCudaStats      与上面平行的 C 结构（定长 double[4]）

IntensityNormalizeDeviceView     驻留 GPU 的结果指针
  templateRoi / dataNormalized / absoluteDifference / l2Difference / statistics
  owner + generation             校验视图是否仍属于当前 workspace
```

支持格式：仅 `CV_8UC1`、`CV_8UC3`。尺寸、类型必须一致，非空。

---

## 4. 模块内部逻辑（CPU / CUDA 共用统计）

```
输入 templateRoi / dataRoi（对齐后的重叠，U8，1 或 3 通道）
        │
        ▼
每通道 256-bin 直方图
        │
        ▼
robustMeanStdFromU8Histogram      ← 只在 host，两边共用
        │
        ├─ 中位数 median（分位 0.5）
        ├─ MAD = |v-median| 的中位数
        ├─ sigma = 1.4826 * MAD
        ├─ 丢掉 |v-median| > 3*sigma 的 bin
        └─ 对内点算 mean / std
        │
        ▼
gain = (dataStd > 1e-9) ? templateStd / dataStd : 1
bias = templateMean - gain * dataMean
        │
        ▼
dataNormalized = saturate(data * gain + bias)     模板不映射
        │
        ▼
absdiff → absoluteDifference
L2：单通道 = abs 转 float；三通道 = sqrt(Σ Δc²)
```

平坦图：`dataStd <= 1e-9` 时 `gain=1`，只校正均值偏移。

---

## 5. 稳健统计 `robustMeanStdFromU8Histogram`

像素数 `n = pixelCount`。分位取值：

```
target = clamp(frac * (n-1), 0, n-1)
累加 hist[0..] 直到 acc > target
```

MAD 目标下标 `(n-1)/2`，在 `|v-median|` 直方图上同样用 `acc > target`。

分支：

| 条件 | mean | std |
|------|------|-----|
| `n <= 0` | 0 | 0 |
| `sigma < 1e-9` | median | 0 |
| 门限内无像素 | median | sigma |
| 否则 | 内点均值 | 内点标准差（总体，`sumSq/count - mean²`） |

`1.4826` 把正态 MAD 换成 σ。大缺陷落在 3σ 外，不进 mean/std，从而不主导 gain/bias。

---

## 6. CPU 结构（`intensityNormalize`）

```
validatePair
连续化 clone（若 step 不紧凑）
split 通道
每通道：
  数直方图 → robustMeanStdFromU8Histogram
  convertTo(CV_8U, gain, bias)     OpenCV saturate_cast
merge → dataNormalized
templateNormalized = reference.clone()
absdiff
单通道：absdiff.convertTo(CV_32F) 作为 L2
三通道：转 float → (t-d)² 按通道相加再 sqrt
```

---

## 7. CUDA 结构

```
IntensityNormalizeCuda
  持有 DeviceWorkspace*
  compute()              H2D → 驻留计算 → D2H；模板用 host memcpy，不再 D2H
  computeResident()      只留在 GPU，返回 DeviceView
  downloadResident()     从 View 拉 dataNormalized / abs / L2 / stats
  workspaceBytes()
  同一实例必须串行；并发晶圆要每线程一个 engine

C API
  intensityNormalizeCompute
  intensityNormalizeReleaseWorkspace
```

生产封装 `intensityNormalizeCuda(engine, mats)` 走 `compute()`（含拷贝）。
`computeResident` 给后续仍在 GPU 上的检出预留，当前管线未用。

### 7.1 DeviceWorkspace

```
dBytes     三块紧凑 U8：template | data（原地写成 normalized）| absdiff
dL2        每像素 float
dHist      256 × 6（template 最多 3 通道 + data 最多 3 通道）
dStats     24 个 double：
           [0:4) tMean  [4:8) tStd  [8:12) dMean
           [12:16) dStd [16:20) gain [20:24) bias
generation 每次 computeResident 递增，使旧 View 失效
```

模板归一化结果不占 GPU 缓冲：它等于输入模板，host 侧 `memcpy`。

### 7.2 Kernel / host 步骤

| 步骤 | 位置 | 作用 |
|------|------|------|
| H2D | host | template、data |
| `interleavedHistogramKernel` | device | 交错 U8 按通道 `atomicAdd` 到 256 bin |
| D2H 直方图 | host | |
| `robustMeanStdFromU8Histogram` | host | 与 CPU 同一函数 |
| H2D `dStats` | host | gain/bias 等 |
| `normalizeAndDifferenceKernel<1/3>` | device | 映射、abs、L2 |
| D2H | host | normalized data、abs、L2、stats |

映射用 FP32，统计仍是 FP64：

```
mapped = clamp(data * (float)gain + (float)bias, 0, 255)
d = (unsigned char)__float2int_rn(mapped)    最近偶数，靠近 OpenCV saturate_cast
abs = |t - d|
l2  = sqrt(Σ delta²)
```

`compute()` 里模板输出直接 `memcpy(templateNormalized, templateRoi)`，不从 GPU 拉回。

View 校验：`owner`、`generation`、宽高通道、各指针必须仍指向当前 workspace，否则 `cudaErrorInvalidValue`。

---

## 8. API 对照

| 逻辑 | CPU | CUDA |
|------|-----|------|
| 高层 | `intensityNormalize(tmpl, data)` | `intensityNormalizeCuda(engine, tmpl, data)` |
| 直方图 | host 遍历 | `interleavedHistogramKernel` |
| 稳健统计 | `robustMeanStdFromU8Histogram` | 同左 |
| 映射 | `Mat::convertTo(gain, bias)` | `normalizeAndDifferenceKernel` |
| 差分 | `absdiff` + split/sqrt | 同一 kernel 写出 |
| 驻留 GPU | 无 | `computeResident` / `downloadResident` |

管线实际调用：

```
CPU : intensityNormalize(out.templateCommon, out.dataCommon)
CUDA: intensityNormalizeCuda(*ctx.intensityCuda, out.templateCommon, out.dataCommon)
```

输入会先保证连续；CUDA 按 `n = w*h`、packed 通道。

---

## 9. 配方（`main.cpp`）

```
normalize_method = Cpu     可改 Cuda / Auto
无独立辐射参数；median / MAD / 3σ / 1.4826 写死在 robustMeanStdFromU8Histogram
```

必须在对齐后的 `templateCommon` / `dataCommon` 上做，不能用未裁黑边的整幅 warped 图。

---

## 10. 与对位、检出的边界

| 模块 | 输入 | 输出用途 |
|------|------|----------|
| Phase / ECC | 二值或灰度 | shift → 重叠 ROI |
| IntensityNormalize | 重叠 ROI 原图像素 | 辐射对齐后的图 + abs / L2 |
| DefectDetect | L2、归一化后的 template/data | 极性、自适应阈值、blob |

模板亮度保持原样，只有 data 被拉到模板的稳健均值/方差上，后面的差分才表示结构差异而不是全局亮度差。
