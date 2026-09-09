# 工程结构与数据流

## 1. 工程定位

c3000 是 Windows x64 上的 C++17/CUDA Die-to-Die 晶圆检测程序。它完成 PNG/XML 配对、die 切割、模板选择、CPU/CUDA 对位、强度归一化、差分缺陷检测以及结果落盘。

主要依赖：

- OpenCV：图像读写、变换、形态学、连通域和 CPU 算法。
- CUDA Runtime：GPU 内存、kernel 调度和设备管理。
- cuFFT：CUDA Phase Correlation。
- MSVC/Visual Studio 2022：工程构建。

## 2. 目录职责

```text
c3000/
├─ include/                 应用层公开类型与接口
│  ├─ die_types.h           Options、DieMap、DieBox、Job
│  ├─ die_cli.h             运行模式与命令行解析接口
│  ├─ die_config.h          默认生产配方接口
│  ├─ die_app.h             作业收集和应用编排
│  ├─ die_map.h             XML/几何映射
│  ├─ die_cut.h             die 切割
│  └─ die_inspect.h         模板、晶圆检测与 patch 评测
├─ src/                     应用层实现
│  ├─ main.cpp              极薄入口：CLI 分发与退出码
│  ├─ die_cli.cpp           参数解析和帮助
│  ├─ die_config.cpp        集中维护算法配方
│  ├─ die_app.cpp           多作业编排
│  ├─ die_map.cpp           XML 解析、分辨率和 die 网格
│  ├─ die_cut.cpp           patch/overlay/CSV 输出
│  ├─ die_inspect.cpp       模板选择、检测循环和结果汇总
│  └─ die_util.cpp          路径、扩展名和 Python 兼容取整
├─ inspect/                 D2D 算法模块
│  ├─ Pipeline.*            统一 CPU/CUDA 检测流水线
│  ├─ PhaseCorrelate.*      CPU/CUDA 相位相关对位
│  ├─ EccAlign.*            CPU/CUDA ECC 平移对位
│  ├─ HistogramThreshold.*  CPU/CUDA 直方图阈值
│  ├─ IntensityNormalize.*  CPU/CUDA 鲁棒强度归一化
│  ├─ DefectDetect.*        极性、阈值、形态学、连通域与过滤
│  ├─ Mssim.*               CPU/CUDA MSSIM；当前仅自检使用
│  └─ SelfCheck.*           合成回归测试和 benchmark
├─ docs/                    架构、构建、测试和优化文档
├─ c3000.vcxproj            x64 Debug/Release 工程
└─ c3000.sln                Visual Studio 解决方案
```

## 3. 顶层调用链

```text
main
 ├─ --help       -> print_usage
 ├─ --self-test  -> runRegressionTests
 ├─ --benchmark  -> runBenchmark
 └─ processing   -> cutdie::run
                    ├─ collect_jobs
                    ├─ load_templates（外部模板只加载一次）
                    ├─ cut_patches
                    │  ├─ parse_die_map
                    │  ├─ detect_lut_width / generate_dies
                    │  └─ patches + die_map.csv + die_overlay.png
                    └─ inspect_wafer
                       ├─ 选择单模板/同坐标模板/相邻模板
                       ├─ runD2DPipeline
                       └─ defects.csv + defect_overlay.png + die_defects/
```

`eval_patches()` 是另一条入口：跳过晶圆切割，直接对已切 patch 做检测，并以 truth 目录中的同名文件计算 TP/FP/FN/TN。

## 4. D2D Pipeline

```text
Template/Data ROI
  -> 统一 CV_8U 与灰度图
  -> Histogram Threshold + Binary
  -> Phase Correlation 或 ECC 对位
  -> 原始 data 亚像素 warp
  -> 有效公共 ROI
  -> 分通道鲁棒强度归一化
  -> Absolute / Signed / L2 Difference
  -> 亮暗极性门控
  -> 自适应阈值
  -> Morphology
  -> Connected Components（一次）
  -> 特征提取与 Blob/极性过滤
  -> candidates / blobMask / overlay
```

二值图只用于对位；最终差分始终基于保留的原始图像和公共有效 ROI。

## 5. 关键数据与所有权

- `Options`：一次作业的路径、后端和检测配方。
- `CutResult`：原始晶圆 `cv::Mat`、die 框和切割产物路径。
- `TemplateCache`：单模板或 `(x,y) -> cv::Mat` 映射；OpenCV Mat 复制为引用计数浅拷贝。
- `PipelineContext`：一次检测会话复用的 CPU/CUDA engine 指针和配方。
- `PipelineOutput`：对位结果、中间图、候选和阶段耗时。
- 每个 CUDA engine 拥有可复用 workspace；同一 engine 不允许并发调用。并行处理时应为每个 worker 创建独立 context/engine。

## 6. 输出约定

| 输出 | 含义 |
|---|---|
| `patches/*.png` | 切出的 die 图像 |
| `die_map.csv` | die 索引、坐标、尺寸、跳过状态和文件名 |
| `die_overlay.png` | 晶圆布局预览 |
| `defects.csv` | 缺陷坐标、极性、面积和差分统计 |
| `defect_overlay.png` | 映射回原始晶圆坐标的缺陷图 |
| `die_defects/*.png` | 有缺陷 die 的局部标注 |
| `eval_report.csv` | patch 评测汇总 |
| `eval_candidates.csv` | patch 候选明细 |

清理旧 patch 时只删除当前图像名前缀且符合 `<stem>_die_x*_y*.png` 的生成文件，不再删除目标目录内所有 PNG。

## 7. 当前边界与技术债

- CPU 路径仍在编译期依赖 CUDA 头和 CUDA translation unit，尚未形成 CPU-only target。
- `die_inspect.cpp` 的 wafer/eval 两条路径仍有 backend context 初始化与单 patch 执行逻辑重复。
- `PipelineOutput` 保留较多全尺寸中间 Mat；生产路径可继续增加 `Minimal/Overlay/Debug` 输出级别以降低峰值内存。
- 相邻模板的共识目前按“命中的邻居数量”判断，还未对候选位置做 IoU/质心聚类投票。
- XML 仍采用轻量字符串解析，适合当前固定格式，但不等价于通用 XML parser。
- benchmark 为内置合成负载；真实产线数据需要另行执行 dataset 回归。
