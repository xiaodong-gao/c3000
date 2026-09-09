# 构建、回归测试与 Benchmark

## 1. 支持环境

当前工程只支持 Windows x64：

- Visual Studio 2022 / MSVC v143
- C++17
- CUDA Toolkit，默认 `12.8`
- OpenCV，默认库后缀 `4140`
- 默认 CUDA 架构 `compute_86,sm_86`

本轮验证环境：

| 项目 | 值 |
|---|---|
| 日期 | 2026-09-09 |
| Visual Studio / MSBuild | VS 2022 17.14 / MSBuild 17.14 |
| CUDA | 12.8 |
| OpenCV | 4.14.0 |
| GPU | NVIDIA RTX A1000 6GB Laptop GPU |
| Compute Capability | 8.6 |
| Driver | 596.47 |

## 2. 构建

在 Visual Studio Developer PowerShell 中：

```powershell
msbuild c3000.sln /m /t:Build /p:Configuration=Release /p:Platform=x64
```

默认值可以覆盖：

```powershell
msbuild c3000.sln /m /t:Build `
  /p:Configuration=Release /p:Platform=x64 `
  /p:CudaVersion=12.8 `
  /p:OpenCVRoot=C:/sdk/opencv/build `
  /p:OpenCVVersion=4140 `
  /p:OpenCVToolset=vc16 `
  /p:CudaCodeGeneration="compute_86,sm_86"
```

`OpenCVRoot` 未显式传入时会先读取 `OpenCV_DIR`，再使用项目中的本机回退值。Debug 链接 `opencv_world<version>d.lib`，Release 链接无 `d` 后缀版本。

初始工程因固定导入 `CUDA 11.8.props/targets` 而在本机报 `MSB4019`；现已改为 `CudaVersion` 属性。工程还移除了实际无法链接 x64 OpenCV/CUDA 的 Win32/x86 配置，以及同时链接 `cudart_static.lib` 和 `cudart.lib` 引发的 `LNK4098`。

Debug CUDA host 信息使用 `/Z7` 写入对象文件，避免 `/m` 构建时多个 NVCC 进程争用同一个 PDB。Debug/x64 与 Release/x64 均已通过并行构建。

## 3. 命令行运行

```powershell
# 帮助
./x64/Release/c3000.exe --help

# 单图
./x64/Release/c3000.exe `
  --image C:/data/0046.png `
  --xml C:/data/0046.xml `
  --out C:/data/output `
  --resolution-um 46 `
  --template-neighbor 2 `
  --align phase-cpu

# 目录批处理
./x64/Release/c3000.exe `
  --dir C:/data/wafers `
  --recursive `
  --out C:/data/output `
  --resolution-um 46

# 外部模板与 CUDA 后端
./x64/Release/c3000.exe `
  --image C:/data/0046.png `
  --xml C:/data/0046.xml `
  --template C:/data/templates `
  --no-template-neighbor `
  --align ecc-cuda `
  --histogram cuda `
  --normalize cuda
```

不带参数时显示帮助。算法默认配方集中在 `src/die_config.cpp`；输入路径必须通过 CLI 指定。

## 4. 回归测试

```powershell
./x64/Release/c3000.exe --self-test
```

本轮结果：

```text
regression: 333 passed, 0 failed
```

覆盖范围：

- CLI 模式、参数覆盖和非法迭代次数。
- Phase CPU/CUDA：零位移、小位移、亚像素和四象限大位移。
- ECC CPU/CUDA：位移恢复、相关系数和不同高斯核的多 workspace 隔离。
- Histogram CPU/CUDA：峰规则、阈值、灰度/BGR 和二值化。
- Intensity Normalize CPU/CUDA：灰度、彩色、平坦图和大块差异。
- MSSIM CPU/CUDA：相同/变化图、map 和 fusion。
- Defect Detect：亮暗极性、形态学、边缘、Blob、大白区、CCL 标签复用和错误尺寸。
- 完整 Pipeline：CPU/CUDA 一致性、稳定性、mask 尺寸和低纹理拒绝。
- 晶圆合成回归：缺陷原图坐标、CSV、整图和单 die overlay。

真实小黑点用例依赖外部数据集。可将 `C3000_TEST_DATA_DIR` 指向包含 `0050_die_x12_y2.png` 和 `0050_die_x0_y12.png` 的 patch 目录；未设置或文件缺失时明确标记 skip，不计入 333 项合成回归。当前整个自检仍要求可用 CUDA 设备。

## 5. Benchmark 方法

```powershell
./x64/Release/c3000.exe --benchmark 10
```

- Release/x64。
- 固定合成输入，尺寸为 256²、512²、1024²；ECC 最大到 512²。
- 预热 1 次，随后 10 次取中位数。
- CUDA 外层计时包含 host/device 传输、workspace 复用和同步。
- 端到端总耗时关闭内部阶段同步；阶段耗时另跑一次采样，避免污染总耗时。
- 每组 benchmark 校验 API 状态；CCL A/B 还逐像素校验两种实现输出一致。

## 6. 结果快照

### 6.1 直接优化项：Connected Components

| 尺寸 | 旧两遍 CCL | 新单遍复用 | 加速比 |
|---:|---:|---:|---:|
| 256² | 0.241 ms | 0.136 ms | 1.77x |
| 512² | 1.226 ms | 0.725 ms | 1.69x |
| 1024² | 6.183 ms | 3.600 ms | 1.72x |

这是同一进程、同一输入下的 A/B 对照。旧路径先生成 labels，再在候选提取中重复 CCL；新路径由候选提取返回同一份 labels。

### 6.2 模块级 CPU/CUDA

| 模块 | 256² CPU/CUDA | 512² CPU/CUDA | 1024² CPU/CUDA |
|---|---:|---:|---:|
| Phase | 1.068 / 0.683 ms, 1.56x | 5.166 / 2.261 ms, 2.29x | 21.619 / 8.964 ms, 2.41x |
| ECC | 5.534 / 0.586 ms, 9.44x | 20.166 / 1.865 ms, 10.81x | — |
| Normalize 1ch | 0.184 / 0.247 ms, 0.74x | 0.952 / 0.890 ms, 1.07x | 5.336 / 3.409 ms, 1.57x |
| Normalize 3ch | 1.718 / 0.396 ms, 4.33x | 6.673 / 1.499 ms, 4.45x | 27.005 / 5.250 ms, 5.14x |
| MSSIM | 0.941 / 0.317 ms, 2.97x | 5.654 / 0.523 ms, 10.80x | 21.017 / 1.326 ms, 15.85x |

256² 单通道 Normalize 的 CUDA 传输/同步成本高于计算收益，CPU 更快；彩色或更大图像才体现 CUDA 优势。

### 6.3 端到端 Pipeline

| 对位 | 尺寸 | CPU | CUDA | 加速比 |
|---|---:|---:|---:|---:|
| Phase | 256² | 3.854 ms | 3.120 ms | 1.24x |
| Phase | 512² | 13.695 ms | 10.137 ms | 1.35x |
| Phase | 1024² | 62.270 ms | 45.889 ms | 1.36x |
| ECC | 256² | 7.871 ms | 2.643 ms | 2.98x |
| ECC | 512² | 28.767 ms | 10.786 ms | 2.67x |

端到端 CUDA 加速比低于纯 CUDA 模块，因为自适应阈值、形态学、Connected Components、候选过滤和 overlay 仍在 CPU，且当前 host API 包含数据传输。

## 7. 性能回归建议

- 固定 GPU、驱动、电源模式、OpenCV 线程数和后台负载后再比较不同提交。
- 稳定性能门禁应至少 30 次采样，并记录 median、p95/MAD、吞吐量和峰值内存。
- 建议仅在“相对退化超过 15% 且绝对退化超过 0.2 ms”时失败，避免微小抖动。
- 内置 benchmark 是开发快照，不应替代真实 die 尺寸、真实缺陷分布和整片吞吐验收。
