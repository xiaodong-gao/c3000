// 从晶圆 PNG 切割 die，再用外部模板做完整 D2D 检出。
// 缺陷画在原始输入大图上，保存为 defect_overlay.png。
// 全部配方参数只在本文件的 main 里设置，不使用命令行参数。

#include "die_app.h"

#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try {
        cutdie::Options opt;

        // ========== 输入 / 输出 ==========
        opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-gray";
        opt.image = "C:/Users/aspiring135/Desktop/datasets/50um-gray/0050.png";
        opt.xml = "C:/Users/aspiring135/Desktop/datasets/50um-gray/0050.xml";
        opt.out.clear();                   // 空则写到 <图像目录>/<图像名>/
        opt.patch_out.clear();             // 空则写到 <out>/patches
        opt.resolution_um = 50.0;          // 每个像素对应多少 um
        opt.keep_skipped = false;          // 是否保存 XML 中标记跳过的 die
        opt.recursive = false;             // 是否递归扫描输入目录
        opt.overlay_preview_px = 8000;     // die 预览图最长边；缺陷原图仍按原始分辨率保存

        // ========== 模板 / 检出开关 ==========
        // 空 + template_neighbor：切割后用本片相邻 die 做 D2D
        opt.template_path.clear();
        opt.template_neighbor = true;
        opt.template_neighbor_min = 2;      // 两个邻居都检出才算缺陷
        opt.inspect = true;

        // ========== patch 评测（非空则不切晶圆） ==========
        opt.eval_patch_dir.clear();
        opt.eval_truth_dir.clear();

        // ========== 后端 ==========
        // AlignMethod: PhaseCuda / PhaseCpu / EccCuda / EccCpu
        // HistogramMethod / NormalizeMethod: Auto / Cpu / Cuda（Auto 跟随对位后端）
        opt.align_method = cutdie::AlignMethod::PhaseCpu;
        opt.histogram_method = cutdie::HistogramMethod::Cpu;
        opt.normalize_method = cutdie::NormalizeMethod::Cpu;

        // ========== 直方图阈值（对位用二值图） ==========
        opt.histogram.gaussianSigma = 2.5f;      // 直方图平滑 sigma
        opt.histogram.peakHeightRatio = 0.05f;   // 峰高相对主峰的比例阈值
        opt.histogram.minPeakDistance = 15;      // 峰之间最小灰度距离
        opt.histogram.minPeakCountRatio = 0.005f;  // 峰像素占比下限
        opt.histogram.defaultThreshold = 125;    // 找不到有效峰时的回退阈值

        // ========== ECC 对位（align_method 为 EccCpu / EccCuda 时生效） ==========
        // motionType 对应 OpenCV findTransformECC / image_alignment.cpp 的 -m：
        // MOTION_TRANSLATION / MOTION_EUCLIDEAN / MOTION_AFFINE / MOTION_HOMOGRAPHY
        opt.ecc.maxIters = 8;         // 最大迭代次数
        opt.ecc.eps = 1e-5;             // 收敛阈值
        opt.ecc.gaussFiltSize = 3;      // 高斯滤波窗口，必须为正奇数
        opt.ecc.motionType = cv::MOTION_EUCLIDEAN;

        // ========== 自适应阈值（融合差分） ==========
        opt.defect.adaptive.blockSize = 31;  // 局部窗口，奇数；<=0 则按图像尺寸自动
        opt.defect.adaptive.k = 4.0;
        opt.defect.adaptive.minAbs = 14.0;   // L2 绝对下限，压掉晶粒噪声

        // ========== 有符号差分极性（data - template） ==========
        opt.defect.signedDiff.T = 14.0;     // +T 亮缺陷；<=0 则按 MAD 自动
        opt.defect.signedDiff.darkT = 10.0; // 暗缺陷：signed < -darkT；0 则与 T 相同

        // ========== 小黑点增强（黑帽） ==========
        opt.defect.darkSpot.enableBlackHat = false;
        opt.defect.darkSpot.kernel = 15;
        opt.defect.darkSpot.minResponse = 8.0;

        // ========== 形态学 ==========
        opt.defect.morph.openSize = 3;
        opt.defect.morph.closeSize = 3;
        opt.defect.morph.dilateSize = 0;
        opt.defect.morph.erodeSize = 0;
        opt.defect.morph.removeNoiseSize = 0;
        opt.defect.morph.darkOpenSize = 0;

        // ========== Blob / 几何过滤 ==========
        opt.defect.blob.minArea = 25;           // 顶边 16x10 黑点，开运算后面积会变小
        opt.defect.blob.maxArea = 0;
        opt.defect.blob.minWidth = 4;
        opt.defect.blob.minHeight = 4;
        opt.defect.blob.maxAspectRatio = 3.0;
        opt.defect.blob.borderMargin = 4;
        opt.defect.blob.borderStripAspect = 2.5;  // 贴边细长条才丢；紧凑贴边黑点保留
        opt.defect.blob.minBrightArea = 400;  // 小亮斑多为邻居缺陷反相；真亮缺陷走 largeWhite

        // ========== 亮/暗缺陷过滤 ==========
        // 0051 是 33x33 patch，真值是暗底白点/块状。D2D 关掉，只走 largeWhite
        opt.defect.filter.keepBright = false;
        opt.defect.filter.keepDark = false;
        opt.defect.filter.keepMixed = false;
        opt.defect.filter.keepUnknown = false;
        opt.defect.filter.minMeanL2 = 10.0;
        opt.defect.filter.minAbsMeanSigned = 8.0;

        // ========== 白点 / 块状（0051：暗底浅小白点 + 片边块状）==========
        // 灰度 threshold=70 抓块状；白帽单独提浅点，避免和亮核并成细长条被长宽比丢掉
        // tophatMin=24、tophatFloor=50：压掉灰度 48–49 的晶粒尾巴，仍保留峰值 >=50 的浅白点
        opt.defect.largeWhite.enable = true;
        opt.defect.largeWhite.threshold = 70.0;
        opt.defect.largeWhite.minArea = 1;
        opt.defect.largeWhite.minFill = 0.20;
        opt.defect.largeWhite.maxAspectRatio = 20.0;
        opt.defect.largeWhite.morphSize = 0;
        opt.defect.largeWhite.maxBackground = 0.0;
        opt.defect.largeWhite.backgroundMargin = 4;
        opt.defect.largeWhite.tophatSize = 7;
        opt.defect.largeWhite.tophatMin = 24.0;
        opt.defect.largeWhite.tophatFloor = 50.0;

        if (opt.resolution_um <= 0.0) {
            throw std::runtime_error("每个像素对应的 um 必须大于 0");
        }
        return cutdie::run(opt);
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << '\n';
        return 1;
    }
}
