# 工程优化报告

## 1. 本轮目标与原则

本轮以不改变公开检测配方和产物格式为前提，优先处理可复现运行、明确 correctness 风险、重复计算和不安全文件清理。所有核心变化都经过 Release 构建、合成回归和 benchmark 验证。

## 2. 已完成改动

### 2.1 入口、配置与构建

- 将原先 `main.cpp` 中的运行开关改为 `--self-test`、`--benchmark [N]` 和 `--help`。
- 将集中配方移动到 `src/die_config.cpp`，入口只负责 CLI 分发和退出码。
- 新增单图、目录、模板、后端、分辨率和 patch 评测等 CLI 覆盖项。
- 移除源码中的个人数据集默认路径；无参数时显示帮助。
- CUDA customization 由固定 11.8 改为可覆盖的 `CudaVersion`，默认 12.8。
- OpenCV 根目录、版本后缀、工具集目录和 CUDA 架构均改为 MSBuild 属性。
- 删除不可工作的 Win32/x86 配置，仅保留真实支持的 x64。
- 去除重复的 `cudart_static.lib` 显式依赖，解决与 CUDA targets 自动链接 `cudart.lib` 的 CRT 冲突警告。
- Debug CUDA host 调试信息改用对象内 `/Z7`，修复并行 NVCC 写同一 PDB 的 `C1041`。
- 新增 `.gitignore`，隔离 VS/CUDA 中间文件和程序生成目录。

### 2.2 Correctness 修复

- Phase Correlation 原先在四个角候选中优先选择“位移模最小”的峰，容易让近原点噪声压过真实大位移峰；现在先选相关值最大峰，仅在相同值时以较小位移打破平局。
- Phase CPU 的小图搜索窗口限制在图像范围内。
- Pipeline 对全前景/全背景二值图拒绝 Phase 对位，避免低纹理输入静默给出伪位移。
- ECC CUDA 高斯系数由全局 constant memory 改为每个 workspace 独立 device buffer，消除不同 `gaussFiltSize` 实例交替调用时的系数污染。
- 缺陷轮廓映射回晶圆时补上 `commonRoi` 偏移，使 contour 与 bbox/centroid 使用同一坐标系。
- `extractDefectCandidates()` 增加 mask、尺寸和类型校验；`drawDefectOverlay()` 正确处理 BGRA 输入。

### 2.3 性能与资源优化

- Connected Components 从两次降为一次：候选提取同时返回 labels，过滤 mask 直接复用。
- `maskFromCandidates()` 用紧凑 label lookup 数组代替逐像素 `unordered_set` 查询。
- 多晶圆外部模板只加载一次；每个作业复用引用计数 Mat。
- `TemplateStore` 持有只读引用，不再复制整张模板 map。
- 邻居模板 MAE 改用 `cv::norm(..., NORM_L1)`，避免显式分配差分 Mat。
- 多模板检测不再无条件 clone 待测 die；仅在尺寸确实需要裁剪时复制。
- patch 输出清理只命中当前图像的生成文件，既缩小删除范围，也避免扫描后重写无关 PNG。

### 2.4 测试与 Benchmark 可用性

- 回归测试由修改源码开关变为一条稳定命令。
- 新增 CLI、Phase 四象限大位移、ECC workspace 隔离、CCL label 复用、错误尺寸和低纹理用例。
- benchmark 拒绝非正迭代次数，检查 CUDA/API/Pipeline 状态。
- 端到端总计时与阶段同步计时分离。
- 新增旧双 CCL 与新单 CCL 的同进程 A/B benchmark，并验证输出逐像素相同。

## 3. 验证结论

| 门禁 | 结果 |
|---|---|
| Debug/x64 与 Release/x64 完整构建 | 通过，无原有 `MSB4019`/`LNK4098`/`C1041` |
| 合成回归 | 333 passed, 0 failed |
| 真实小黑点回归 | 数据集缺失，明确跳过 |
| CCL A/B 输出 | 逐像素一致 |
| CCL A/B 性能 | 1.69x–1.77x |
| CLI help/非法参数退出码 | 通过 |

完整环境与性能表见 [build-test-benchmark.md](build-test-benchmark.md)。

## 4. 未在本轮冒险修改的算法行为

以下问题需要真实数据 golden set 或更明确的业务规则，未直接改变：

- 相邻模板目前只统计“哪些邻居出现任意候选”，没有验证多个邻居是否在同一位置发现同一缺陷。
- 不同尺寸 template/data 采用左上裁剪；改为中心裁剪需要同步维护坐标偏移。
- 16U/32F 到 8U 的缩放策略仍较简单，真实动态范围策略应由数据规格确定。
- Phase 仅有“非均匀二值图”门禁，尚未引入 PSR、最小 response 和最大业务位移参数。
- `PipelineOutput` 默认保留大量调试中间图，尚未按产线/调试场景裁剪。

## 5. 推荐后续顺序

1. 建立小型、可入库的匿名真实数据集及 golden candidates/masks。
2. 将 CPU core、CUDA backend、应用、CPU tests、CUDA tests、benchmark 拆成独立 target。
3. 抽取拥有 engine 的 `PipelineSession` 和共享 `inspect_one_patch()`，消除 wafer/eval 重复逻辑。
4. 为邻居候选做空间聚类或 IoU 投票，并把可用邻居不足视为明确状态。
5. 增加 `ArtifactLevel { Minimal, Overlay, Debug }`，降低生产路径内存和图像绘制开销。
6. 输出结构化 benchmark CSV/JSON，并在固定 GPU runner 上做定时性能回归。
7. 用正式 XML parser 和必填几何字段校验替换字符串扫描。
