# IntensityNormalize 回归测试报告

测试日期：2026-09-11  
测试平台：NVIDIA RTX A1000 Laptop GPU（Compute Capability 8.6）、CUDA 12.8、OpenCV 4.14.0、Visual Studio 2022 v143

## 本次实现范围

按《IntensityNormalize Algorithm Specification.md》的 P0 优先级完成：

- `Offset`、`Gain`、`GainOffset`、`MeanStd` 可配置策略；
- `BackgroundSubtract` 和 `LocalMean` 空间局部归一化；
- `CV_8U`、`CV_16U`、`CV_32F`，1–4 通道；
- 可选 `CV_8UC1` Valid Mask；
- Gain/Offset 上下限、输出范围和裁剪计数；
- `StdDev < epsilon` 时的安全 fallback；
- `CV_32F` NaN/Inf 拒绝；
- CPU Golden Reference 与 CUDA 实现；
- CPU/CUDA 统一的 round-to-nearest-even 舍入；
- CUDA Stream 参数、融合 Normalize/Clamp/Difference Kernel、复用型 Workspace；
- 兼容旧的 robust API 和 resident view API。

空间背景采用 Mask-aware、Replicate Border 的可分离 Box Blur，窗口半径由 `localRadius` 配置。`BackgroundSubtract` 使用 `D' = D - B_D + B_T` 保持 Template 灰度域；`LocalMean` 对两幅图分别计算 `I-B(x,y)`，整数图增加中点锚值以保留正负残差。

`Robust` 当前完整支持 `CV_8U`。规范中的 Percentile、Histogram Matching、BackgroundDivide、LocalMeanStd、Gamma、Log、ZScore 等其余 P1–P3 策略仅保留枚举，调用时明确返回“不支持”，不会静默退化成其他算法。

## 回归结果

Release 与 Debug 均编译、运行成功。自动化用例覆盖：

- Template == Data；
- 常量 Offset；
- Gain；
- Gain + Offset；
- Hot Pixel / Robust；
- Defect Mask 与缺陷保留；
- 零方差 fallback；
- Saturation/Clipping；
- NaN/Inf；
- 策略隔离；
- legacy resident API 与失效 view 检查；
- 3 种数据深度 × 1/3 通道 × 4 种 P0 方法的 CPU/GPU 对齐；
- `CV_8UC3 Robust` CPU/GPU 对齐。
- `BackgroundSubtract / LocalMean` 在三种深度、1/3 通道下的 CPU/GPU 对齐。

结果：`53 passed, 0 failed`（包含 non-blocking CUDA Stream 用例）。

CPU/GPU 输出精度：

| 类型 | 最大绝对误差 | 规范要求 | 结果 |
|---|---:|---:|---|
| CV_8U | 0 | ≤ 1 | 通过 |
| CV_16U | 0 | 本工程阈值 ≤ 1 | 通过 |
| CV_32F | 0 | 相对/绝对容差 | 通过 |

全局缺陷保留用例：背景 MAE 为 `0`，人工缺陷归一化后平均对比度为 `149`。

空间梯度用例：归一化前背景 MAE 为 `14.1991`；`BackgroundSubtract` 降至 `0.2625`，`LocalMean` 降至 `0.4729`，人工缺陷对比度均保持为 `65.4286`。

CUDA Compute Sanitizer：`ERROR SUMMARY: 0 errors`。

## Release 端到端基准

统计、参数计算、Normalize、Clamp、Difference、H2D/D2H 均计入。单次运行结果如下（机器负载会影响绝对耗时）：

| ROI | CPU (ms) | GPU (ms) | Speedup | Workspace (MiB) |
|---|---:|---:|---:|---:|
| 256×256 | 0.523 | 0.436 | 1.201× | 6.739 |
| 512×512 | 2.194 | 1.068 | 2.054× | 7.489 |
| 1024×1024 | 8.935 | 3.504 | 2.550× | 12.739 |
| 2048×2048 | 33.942 | 11.725 | 2.895× | 36.739 |
| 4096×4096 | 139.052 | 43.076 | 3.228× | 132.739 |

`BackgroundSubtract`、Box Radius 7、`CV_8UC1`：

| ROI | CPU (ms) | GPU (ms) | Speedup | Workspace (MiB) |
|---|---:|---:|---:|---:|
| 256×256 | 5.474 | 0.940 | 5.821× | 132.800 |
| 512×512 | 22.666 | 2.413 | 9.393× | 137.008 |
| 1024×1024 | 91.834 | 7.592 | 12.096× | 164.008 |
| 2048×2048 | 375.180 | 26.845 | 13.976× | 272.008 |
| 4096×4096 | 1573.792 | 103.661 | 15.182× | 704.008 |

## 复现命令

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe' `
  'test-IntensityNormalize.sln' /t:Rebuild /p:Configuration=Release /p:Platform=x64

$env:PATH = 'C:\development\opencv-4.14.0-windows\build\x64\vc16\bin;' + $env:PATH
& '.\x64\Release\test-IntensityNormalize.exe'

& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\compute-sanitizer.bat' `
  --tool memcheck --error-exitcode 9 '.\x64\Release\test-IntensityNormalize.exe'
```

## 尚待生产验收的项目

- 长时间稳定性/压力测试尚未执行；
- 当前基准矩阵是所有规范 ROI 尺寸下的默认 `CV_8UC1 GainOffset`，尚未扩展成所有数据类型与所有策略的笛卡尔积；
- Host-facing API 为返回完整诊断信息，会在统计量回传和最终结果回传处同步；若接入纯 GPU 检测流水线，应继续增加通用 device-pointer 异步入口；
- Background 空间工作区为换取确定性双精度背景和 CPU/GPU 严格对齐而保留四个 FP64 中间平面；4096×4096 C1 实测总工作区约 704 MiB，生产接入前可进一步做分块/环形缓冲优化；
- 其余 P1–P3 算法需按实际场景逐项实现，并分别增加缺陷保留和 CPU/GPU 对齐用例。
