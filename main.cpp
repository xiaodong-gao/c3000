// 读取对齐 XML 的圆形 ROI，整圆直接送入 SingleDie 检出。不切 die patch。
// 缺陷画在圆形 ROI 上，保存为 defect_overlay.png。
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
        opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-no-partern-wafer";
        opt.image = "C:/Users/aspiring135/Desktop/datasets/50um-no-partern-wafer/50um-nopatern.png";
        opt.xml = "C:/Users/aspiring135/Desktop/datasets/50um-no-partern-wafer/50um-nopatern.xml";
        opt.out.clear();                   // 空则写到 <图像目录>/<图像名>/
        opt.patch_out.clear();             // SingleDie 模式不切 patch，此路径无效
        opt.resolution_um = 50.0;          // 每个像素对应多少 um
        opt.keep_skipped = false;          // 是否保存 XML 中标记跳过的 die
        opt.recursive = false;             // 是否递归扫描输入目录
        opt.overlay_preview_px = 8000;     // 圆形 ROI 预览图最长边

        // ========== 检出开关 ==========
        opt.template_path.clear();
        opt.template_neighbor = false;
        opt.template_neighbor_min = 1;
        opt.inspect = true;
        opt.singledie = true;              // 只切圆形 ROI，直接进 SingleDie

        // ========== SingleDie（读对齐 XML，只截圆形 ROI）==========
        // 目标：把圆内所有亮/暗缺陷都检出来（含 1px、划痕、大块）。
        // 背景核要明显大于缺陷，避免大缺陷被吃进背景。
        opt.singledie_normalize.background.method = singledie::BackgroundMethod::Gaussian;
        opt.singledie_normalize.background.kernelSize = 201;
        opt.singledie_normalize.background.sigma = 0.0;
        opt.singledie_normalize.method = singledie::IntensityMethod::Divide;
        opt.singledie_normalize.minBackground = 1e-6f;
        opt.singledie_normalize.robustPercentile = true;

        opt.singledie_detect.enableBright = true;
        opt.singledie_detect.enableDark = true;
        opt.singledie_detect.brightThreshold = 0.025f;
        opt.singledie_detect.darkThreshold = 0.025f;
        opt.singledie_detect.morphologyKernel = 3;
        opt.singledie_detect.openIterations = 0;     // 开运算会吃掉 1–4px 小缺陷
        opt.singledie_detect.closeIterations = 1;    // 只补断裂，不删点
        opt.singledie_detect.minArea = 1;            // XML DSize 从 0 um 起，50um/px 对应 1px
        opt.singledie_detect.maxArea = 10000000;     // 不限制大缺陷
        opt.singledie_detect.borderMargin = 0;       // 圆边缺陷不要当贴边丢掉
        opt.singledie_detect.enableBlob = false;
        opt.singledie_detect.keepResponseMaps = false;
        opt.singledie_detect.maxAspectRatio = 0.0f;  // 0 关闭，保留划痕/细长缺陷
        opt.singledie_detect.minFillRatio = 0.0f;

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
        opt.defect.adaptive.minAbs = 18.0;   // L2 下限，压掉网格纹理

        // ========== 有符号差分极性（data - template） ==========
        opt.defect.signedDiff.T = 16.0;     // +T 亮缺陷；<=0 则按 MAD 自动
        opt.defect.signedDiff.darkT = 14.0; // 暗缺陷：signed < -darkT；0 则与 T 相同

        // ========== 小黑点增强（黑帽） ==========
        opt.defect.darkSpot.enableBlackHat = true;
        opt.defect.darkSpot.kernel = 15;
        opt.defect.darkSpot.minResponse = 12.0;

        // ========== 形态学 ==========
        opt.defect.morph.openSize = 0;       // 小黑点不要开运算，3x3 会吃掉
        opt.defect.morph.closeSize = 3;
        opt.defect.morph.dilateSize = 0;
        opt.defect.morph.erodeSize = 0;
        opt.defect.morph.removeNoiseSize = 0;
        opt.defect.morph.darkOpenSize = 0;

        // ========== Blob / 几何过滤 ==========
        // 0046 真值是图案上的紧凑圆黑点（约 8–120 px），不是片边白区
        opt.defect.blob.minArea = 8;
        opt.defect.blob.maxArea = 250;       // 丢掉片外大块黑区
        opt.defect.blob.minWidth = 3;
        opt.defect.blob.minHeight = 3;
        opt.defect.blob.maxAspectRatio = 4.0;
        opt.defect.blob.borderMargin = 4;
        opt.defect.blob.borderStripAspect = 2.5;  // 贴边细长条才丢；紧凑贴边黑点保留
        opt.defect.blob.minBrightArea = 400;

        // ========== 亮/暗缺陷过滤 ==========
        // 0046 名单是图案上的圆黑点。D2D 关掉：片边对位会打出大量碎暗斑
        opt.defect.filter.keepBright = false;
        opt.defect.filter.keepDark = false;
        opt.defect.filter.keepMixed = false;
        opt.defect.filter.keepUnknown = false;
        opt.defect.filter.minMeanL2 = 14.0;
        opt.defect.filter.minAbsMeanSigned = 12.0;

        // ========== 白点 / 大白块（暗场按绝对灰度，1px 到整块白区都收）==========
        opt.defect.largeWhite.enable = true;
        opt.defect.largeWhite.threshold = 120.0;   // 暗场小白点；晶圆亮边约 60，120 以上才收
        opt.defect.largeWhite.minArea = 1;         // 含漏检的 1x1 白点
        opt.defect.largeWhite.minFill = 0.0;       // 1x1 不卡填充率
        opt.defect.largeWhite.maxAspectRatio = 2.5;
        opt.defect.largeWhite.morphSize = 0;       // 开运算会吃掉 1px
        opt.defect.largeWhite.maxBackground = 0.0;
        opt.defect.largeWhite.backgroundMargin = 4;
        opt.defect.largeWhite.tophatSize = 0;
        opt.defect.largeWhite.tophatMin = 0.0;
        opt.defect.largeWhite.tophatFloor = 0.0;
        opt.defect.largeWhite.tophatMinArea = 0;
        opt.defect.largeWhite.tophatMaxArea = 0;
        opt.defect.largeWhite.tophatMinFill = 0.0;
        opt.defect.largeWhite.tophatMaxAspect = 0.0;
        opt.defect.largeWhite.tophatStreetBand = 0;
        opt.defect.largeWhite.tophatBorderMargin = 0;
        opt.defect.largeWhite.tophatMinPeak = 0.0;
        opt.defect.largeWhite.isolatedMaxArea = 16;      // 底部贴边小白点常是 3x3 簇
        opt.defect.largeWhite.isolatedThreshold = 80.0;  // 过低会和亮边连成大块；80 保住孤立白点
        opt.defect.largeWhite.isolatedInnerMargin = 16;  // 只滤圆掩膜造成的假 1x1，孤立通路不内缩

        // ========== 本颗暗粒子（只收对比够、比较圆的黑点）==========
        opt.defect.darkRegion.enable = true;
        opt.defect.darkRegion.blackhatSize = 15;
        opt.defect.darkRegion.blackhatMin = 16.0;
        opt.defect.darkRegion.belowMedian = 16.0;
        opt.defect.darkRegion.grayMin = 90.0;
        opt.defect.darkRegion.grayMax = 220.0;
        opt.defect.darkRegion.minArea = 16;
        opt.defect.darkRegion.maxArea = 180;
        opt.defect.darkRegion.minFill = 0.40;
        opt.defect.darkRegion.maxAspectRatio = 2.2;
        // 浅一点的小黑点：闭运算拼块，躲开划片槽和贴边；小面积还须黑帽更强
        opt.defect.darkRegion.closeSize = 3;
        opt.defect.darkRegion.faintBlackhatMin = 14.0;
        opt.defect.darkRegion.faintBelowMedian = 10.0;
        opt.defect.darkRegion.faintMinArea = 48;
        opt.defect.darkRegion.faintMinFill = 0.42;
        opt.defect.darkRegion.faintMaxAspect = 2.0;
        opt.defect.darkRegion.streetBand = 8;
        opt.defect.darkRegion.borderMargin = 6;
        opt.defect.darkRegion.faintSmallMinArea = 19;
        opt.defect.darkRegion.faintSmallMaxArea = 21;
        opt.defect.darkRegion.faintSmallMinFill = 0.40;
        opt.defect.darkRegion.faintSmallMaxAspect = 2.0;
        opt.defect.darkRegion.faintSmallHatMin = 24.0;
        // 片边浅小点：只对上下切割边，且只收靠近白边的点，避免左右片边过检
        opt.defect.darkRegion.edgeBrightMin = 2500;
        opt.defect.darkRegion.edgeDarkMin = 2000;
        opt.defect.darkRegion.edgeGrayMin = 90.0;
        opt.defect.darkRegion.edgeGrayMax = 220.0;
        opt.defect.darkRegion.edgeBlackhatMin = 16.0;
        opt.defect.darkRegion.edgeBelowMedian = 10.0;
        opt.defect.darkRegion.edgeMinArea = 10;
        opt.defect.darkRegion.edgeMaxArea = 120;
        opt.defect.darkRegion.edgeMinFill = 0.34;
        opt.defect.darkRegion.edgeMaxAspect = 2.8;
        opt.defect.darkRegion.edgeNearCut = 40;
        opt.defect.darkRegion.edgeBlackFrac = 0.70;

        if (opt.resolution_um <= 0.0) {
            throw std::runtime_error("每个像素对应的 um 必须大于 0");
        }
        return cutdie::run(opt);
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << '\n';
        return 1;
    }
}
