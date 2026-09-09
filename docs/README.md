# c3000 工程文档

本目录记录当前工程结构、构建方式、回归测试、性能基准和本轮优化。建议按下面顺序阅读：

1. [architecture.md](architecture.md)：模块边界、调用链、数据流和输出。
2. [build-test-benchmark.md](build-test-benchmark.md)：环境配置、构建、运行、回归与 benchmark 结果。
3. [optimization-report.md](optimization-report.md)：已完成优化、验证依据和后续路线。
4. [KLA-style Die-to-Die CUDA C++ Implementation Specification.md](KLA-style%20Die-to-Die%20CUDA%20C++%20Implementation%20Specification.md)：原始算法实现规格。

## 快速开始

在 Visual Studio Developer PowerShell 中执行：

```powershell
msbuild c3000.sln /m /t:Build /p:Configuration=Release /p:Platform=x64
./x64/Release/c3000.exe --self-test
./x64/Release/c3000.exe --benchmark 10
```

查看完整命令行：

```powershell
./x64/Release/c3000.exe --help
```

工程只保留可工作的 `x64` 配置；CUDA、OpenCV 版本和 GPU 架构均可通过 MSBuild 属性覆盖，详见构建文档。
