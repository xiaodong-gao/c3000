# EccAlign 逻辑结构

本文对应现有实现，不讨论改法。
职责：估计 data 相对 template 的位移 `(dx, dy)`（+x 右、+y 下），约定与相位对位、`shiftImage` 一致。
对齐图像必须是同尺寸 `CV_8UC1` 灰度图。不要用直方图二值图：梯度稀疏时欧氏模型容易 NaN。

---

## 1. 代码结构（文件 → 职责）

```
ams/
  EccAlign.hpp     CPU 参数 / 结果 / eccShift()
  EccAlign.cpp     调 OpenCV findTransformECC，从 warp 取出平移
  EccAlign.cuh     CUDA 参数 / 结果 / C API / 类 EccAlignCuda
  EccAlign.cu      按 OpenCV ECC 实现 MOTION_TRANSLATION / MOTION_EUCLIDEAN

调用方
  app/die_types.h           Options.ecc（EccAlignParams）
  app/die_inspect.cpp       AlignMethod::EccCpu / EccCuda → 填 PipelineContext
  die2die/Pipeline.cpp      灰度图上算 shift，再 shiftImage
  main.cpp                  配方：maxIters / eps / gaussFiltSize / motionType
```

两套入口、同一套 ECC 数学（CUDA 只覆盖平移和欧氏）：

| 入口 | 实现 | 运动模型 |
|------|------|----------|
| `eccShift()` | OpenCV `findTransformECC` | TRANSLATION / EUCLIDEAN / AFFINE / HOMOGRAPHY |
| `EccAlignCuda::compute()` | `EccAlign.cu` | TRANSLATION(0) / EUCLIDEAN(1) |

---

## 2. 在 D2D 管线中的位置

```
preparePair()                         转灰度（OpenCV）
        │
        ├─ HistogramThreshold         二值图 → 仅 Phase 用
        │
        ▼
align == Ecc
        │
        ├─ EccCpu  → eccShift(templateGray, dataGray, eccParams)
        └─ EccCuda → eccCuda->compute(templateGray, dataGray, w, h)
        │
        ▼
out.shift = (dx, dy)                  warp 的 tx/ty
out.response = cc / rho
        │
        ▼
shiftImage(data, shift)               与 WARP_INVERSE_MAP 同约定
        → 重叠 ROI → 归一化 → 差分检出
```

`AlignMethod`：

```
EccCpu   → PipelineBackend::Cpu  + PipelineAlign::Ecc
EccCuda  → PipelineBackend::Cuda + PipelineAlign::Ecc
Phase*   → 不走 EccAlign，走相位对位（二值图）
```

`die_inspect.cpp` 在需要 CUDA 对位时构造 `EccAlignCuda`，把 `opt.ecc` 拷到 `EccCudaParams`。

---

## 3. 数据类型

CPU（`EccAlign.hpp`）：

```
EccAlignParams
  maxIters        最大迭代，默认 100；配方里常用 8
  eps             |Δrho| 收敛阈值，默认 1e-5
  gaussFiltSize   高斯窗口，必须为正奇数，默认 5；配方里常用 3
  motionType      OpenCV 运动模型，默认 MOTION_EUCLIDEAN

EccAlignResult
  shift           (dx, dy) = warp(0,2), warp(1,2)
  cc              增强相关系数 rho
  ok              是否收敛
  error           失败原因
```

CUDA（`EccAlign.cuh`）：

```
EccCudaParams
  maxIters / eps / gaussFiltSize
  motionType      0 = TRANSLATION，1 = EUCLIDEAN（其它值直接 InvalidValue）

EccCudaResult
  dx, dy          同 CPU 的 shift
  cc              rho
  iterations      实际迭代次数
  ok              1 成功 / 0 未收敛（cudaError 仍可能是 Success）
```

位移约定（CPU 注释，CUDA 同样遵守）：

```
OpenCV warp 为 2x3，WARP_INVERSE_MAP：dst(x) = src(x + tx, y + ty)
shiftImage(src, s) 使用同一个 s = (tx, ty)
```

---

## 4. 模块内部逻辑

ECC 最大化增强相关系数 rho（零均值后的归一化互相关），迭代更新 warp。

```
输入 template / data（U8 灰度，同尺寸）
        │
        ▼
转 float + 高斯模糊（sigma=0 的 OpenCV 固定核，见下）
        │
        ▼
对 data 求梯度 gx, gy（中心差分，BORDER_REFLECT_101）
        │
        ▼
warp 初值 = 单位阵
  TRANSLATION : [1 0 tx; 0 1 ty]
  EUCLIDEAN   : [cosθ  -sinθ  tx; sinθ  cosθ  ty]
        │
        ▼
for iter = 1 .. maxIters  while |rho - lastRho| >= eps
        │
        ├─ warpAffine(data, gx, gy)     INTER_LINEAR + WARP_INVERSE_MAP
        ├─ 有效掩膜：圆整后的 (sx,sy) 落在图内
        ├─ 在掩膜内算均值，再算零均值 rho
        ├─ 投影 Jacobian / Hessian（float 精度，对齐 OpenCV Mat::inv）
        ├─ lambda、参数增量
        └─ 更新 warp
        │
        ▼
输出 (tx, ty) = (h02, h12)，cc = rho
```

OpenCV 对 Hessian 按**全图像素**累加（掩膜外 Jacobian 仍计入）；掩膜外 template 零均值值为 0。CUDA `projKernel` 按此实现。

---

## 5. CPU 结构（`eccShift`）

```
校验
  非空、同尺寸、CV_8UC1
  gaussFiltSize 为正奇数
  motionType ∈ {TRANSLATION, EUCLIDEAN, AFFINE, HOMOGRAPHY}

convertTo CV_32F
warp = I（HOMOGRAPHY 为 3x3，其余 2x3）
criteria = COUNT | EPS，maxIters / eps

findTransformECC(templateF, dataF, warp, motionType, criteria, noArray, gaussFiltSize)

shift.x = warp(0,2)
shift.y = warp(1,2)
cc = 返回的 rho
异常 → ok=false，写入 error
```

CPU 的 AFFINE / HOMOGRAPHY 只在 OpenCV 路径存在；CUDA 不实现。生产配方当前是 `MOTION_EUCLIDEAN`。

---

## 6. CUDA 结构

```
EccAlignCuda
  持有 EccCudaParams + DeviceWorkspace*
  compute() → computeDevice(workspace, ...)
  析构释放 workspace

C API（无对象时用 defaultWorkspace）
  eccAlignCompute
  eccAlignReleaseWorkspace
  eccCudaDefaultParams    maxIters=100, eps=1e-5, gauss=5, motionType=1
```

### 6.1 DeviceWorkspace

按图像尺寸复用：

```
dTmplU8 / dDataU8     输入 U8
dTmpl / dData         float 原图
dTmp                  高斯中间缓冲
dGx / dGy             data 梯度
dWarped / dGxW / dGyW 当前 warp 后的图与梯度
dMask                 有效像素
dRed                  ReducePack 归约
```

float 缓冲一次性分配 `n * 8`，再切片。`gaussSize` 缓存已上传的核，避免每轮 `MemcpyToSymbol`。

### 6.2 Kernel / host 步骤

| 步骤 | 位置 | 作用 |
|------|------|------|
| `u8ToFloatKernel` | device | U8 → float |
| `uploadGauss` + 水平/垂直 blur | host+device | 与 OpenCV `GaussianBlur(sigma=0)` 对齐 |
| `gradientKernel` | device | gx/gy = 0.5 * (右-左, 下-上)，`reflect101` |
| `warpAffineKernel` | device | 逆映射双线性；mask 用 `cvRound(sx/sy)` 是否在图内 |
| `sampleOpenCvLinear` | device | `INTER_TAB_SIZE=32`：`round(s*32)` 再 5 bit 权重 |
| `meanKernel` | device | 掩膜内 count / sumW / sumT / 二阶矩 |
| `centerImageMoments` | host | 零均值二阶矩，算 rho |
| `projKernel` | device | Jacobian 投影；欧氏模型多一列旋转 |
| `invert2x2` / `invert3x3` | host | Hessian 求逆（先转 float） |
| 更新 warp | host | 平移只改 tx/ty；欧氏再更新 θ |

高斯核（`sigma=0`，奇数 ksize≤7）用 OpenCV 固定表，不是 `exp` 公式：

```
ksize=1 : [1]
ksize=3 : [0.25, 0.5, 0.25]
ksize=5 : [0.0625, 0.25, 0.375, 0.25, 0.0625]
ksize=7 : [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
ksize>7 : 0.3*((ksize-1)*0.5-1)+0.8 的高斯，再归一化
```

CUDA `gaussFiltSize` 上限 `2*kMaxRadius+1 = 15`。

### 6.3 迭代更新

**TRANSLATION（2 参数 tx, ty）**

```
J = [gx, gy]
H = JᵀJ（2×2）
投影 warped / template 到 J
lambda = (||w||² - …) / (<t,w> - …)
若分母 ≤ 0 → 失败，cc=-1
(tx, ty) += H⁻¹ (lambda * projT - projW)
```

**EUCLIDEAN（3 参数 θ, tx, ty）**

```
hatX = -sinθ·x - cosθ·y
hatY =  cosθ·x - sinθ·y
j0 = gx*hatX + gy*hatY
J = [j0, gx, gy]
H = JᵀJ（3×3）
增量 (d0, d1, d2)
θ' = d0 + asin(clamp(h10, -1, 1))
tx += d1
ty += d2
[h00 h01; h10 h11] = [cosθ'  -sinθ'; sinθ'  cosθ']
```

对应 OpenCV `update_warping_matrix_ECC` 的 `MOTION_EUCLIDEAN`。

### 6.4 失败（`ok=0`，cudaError 仍为 Success）

```
有效像素 mean.count < 16
rho 非有限
Hessian 奇异
lambda 分母 ≤ 0
motionType 不是 0/1，或 gauss 非法 → cudaErrorInvalidValue
```

循环条件与 OpenCV 相同：`iter <= maxIters && |rho-lastRho| >= eps`。
`lastRho` 初值为 `-eps`，第一次一定进入。

---

## 7. API 对照

| 逻辑 | CPU | CUDA |
|------|-----|------|
| 计算位移 | `eccShift(tmpl, data, params)` | `EccAlignCuda::compute(tmpl, data, w, h, result)` |
| 位移 | `result.shift` | `(result.dx, result.dy)` |
| 相关系数 | `result.cc` | `result.cc` |
| 成功 | `result.ok` | `result.ok != 0` |
| 运动模型 | 四种 OpenCV 模型 | 仅 0/1 |

管线实际调用：

```
CPU : eccShift(out.templateGray, out.dataGray, ctx.eccParams)
CUDA: ctx.eccCuda->compute(templateGray.data, dataGray.data, w, h, ecc)
```

要求 `.data` 连续，`n = w*h`。

---

## 8. 配方（`main.cpp` → `opt.ecc`）

```
maxIters      = 8
eps           = 1e-5
gaussFiltSize = 3          正奇数
motionType    = cv::MOTION_EUCLIDEAN
align_method  = PhaseCpu   改成 EccCpu / EccCuda 才走本模块
```

`motionType` 与 OpenCV `samples/cpp/image_alignment.cpp` 的 `-m` 一致。
CUDA 若配 AFFINE / HOMOGRAPHY 会 `cudaErrorInvalidValue`。

---

## 9. 与 HistogramThreshold / Phase 的边界

| 模块 | 输入 | 输出用途 |
|------|------|----------|
| HistogramThreshold | 灰度 | 二值图，给 Phase |
| EccAlign | 灰度 | shift，给 `shiftImage` |
| PhaseCorrelate | 二值 | shift，给 `shiftImage` |

三者位移约定相同。Ecc 与 Phase 在管线里互斥，由 `PipelineAlign` 选择。
