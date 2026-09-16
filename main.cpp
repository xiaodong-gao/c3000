// 同一工程内切换 Single Die / Die-to-Die。
// 改 kInspectMode 后，main 只加载对应配方并调用 cutdie::run。
// 全部配方参数只在本文件设置，不使用命令行参数。

#include "die_app.h"

#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

void apply_common_backend(cutdie::Options& opt) {
    // AlignMethod: PhaseCuda / PhaseCpu / EccCuda / EccCpu
    // HistogramMethod / NormalizeMethod: Auto / Cpu / Cuda（Auto 跟随对位后端）
    opt.align_method = cutdie::AlignMethod::PhaseCpu;
    opt.histogram_method = cutdie::HistogramMethod::Cpu;
    opt.normalize_method = cutdie::NormalizeMethod::Cpu;

    opt.histogram.gaussianSigma = 2.5f;
    opt.histogram.peakHeightRatio = 0.05f;
    opt.histogram.minPeakDistance = 15;
    opt.histogram.minPeakCountRatio = 0.005f;
    opt.histogram.defaultThreshold = 125;

    opt.ecc.maxIters = 8;
    opt.ecc.eps = 1e-5;
    opt.ecc.gaussFiltSize = 3;
    opt.ecc.motionType = cv::MOTION_EUCLIDEAN;
}

// 有图形晶圆：切割 die patch，用相邻 die 做 D2D。
void apply_d2d_recipe(cutdie::Options& opt) {
    opt.mode = cutdie::InspectMode::DieToDie;

    opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-0804";
    opt.image = "C:/Users/aspiring135/Desktop/datasets/50um-0804/0804-slot1.png";
    opt.xml = "C:/Users/aspiring135/Desktop/datasets/50um-0804/0804-slot1.xml";
    opt.out.clear();
    opt.patch_out.clear();
    opt.resolution_um = 50.0;
    opt.keep_skipped = false;
    opt.recursive = false;
    opt.overlay_preview_px = 8000;

    opt.template_path.clear();
    opt.template_neighbor = true;
    opt.template_neighbor_min = 2;
    opt.inspect = true;

    opt.eval_patch_dir.clear();
    opt.eval_truth_dir.clear();

    apply_common_backend(opt);

    opt.defect.adaptive.blockSize = 31;
    opt.defect.adaptive.k = 4.0;
    opt.defect.adaptive.minAbs = 18.0;

    opt.defect.signedDiff.T = 16.0;
    opt.defect.signedDiff.darkT = 14.0;

    opt.defect.darkSpot.enableBlackHat = true;
    opt.defect.darkSpot.kernel = 15;
    opt.defect.darkSpot.minResponse = 12.0;

    opt.defect.morph.openSize = 0;
    opt.defect.morph.closeSize = 3;
    opt.defect.morph.dilateSize = 0;
    opt.defect.morph.erodeSize = 0;
    opt.defect.morph.removeNoiseSize = 0;
    opt.defect.morph.darkOpenSize = 0;

    opt.defect.blob.minArea = 8;
    opt.defect.blob.maxArea = 250;
    opt.defect.blob.minWidth = 3;
    opt.defect.blob.minHeight = 3;
    opt.defect.blob.maxAspectRatio = 4.0;
    opt.defect.blob.borderMargin = 4;
    opt.defect.blob.borderStripAspect = 2.5;
    opt.defect.blob.minBrightArea = 400;

    opt.defect.filter.keepBright = false;
    opt.defect.filter.keepDark = false;
    opt.defect.filter.keepMixed = false;
    opt.defect.filter.keepUnknown = false;
    opt.defect.filter.minMeanL2 = 14.0;
    opt.defect.filter.minAbsMeanSigned = 12.0;

    opt.defect.largeWhite.enable = true;
    opt.defect.largeWhite.threshold = 230.0;
    opt.defect.largeWhite.minArea = 7000;
    opt.defect.largeWhite.minFill = 0.35;
    opt.defect.largeWhite.maxAspectRatio = 1.6;
    opt.defect.largeWhite.morphSize = 0;
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
    opt.defect.largeWhite.isolatedMaxArea = 0;
    opt.defect.largeWhite.isolatedThreshold = 0.0;
    opt.defect.largeWhite.isolatedInnerMargin = 16;

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
}

// 无图形晶圆：只切圆形 ROI，整圆送入 SingleDie。
void apply_singledie_recipe(cutdie::Options& opt) {
    opt.mode = cutdie::InspectMode::SingleDie;

    opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-0804";
    opt.image = "C:/Users/aspiring135/Desktop/datasets/50um-0804/0804-slot3.png";
    opt.xml = "C:/Users/aspiring135/Desktop/datasets/50um-0804/0804-slot3.xml";
    opt.out = "C:/Users/aspiring135/Desktop/datasets/50um-0804";
    opt.patch_out.clear();
    opt.resolution_um = 50.0;
    opt.keep_skipped = false;
    opt.recursive = false;
    opt.overlay_preview_px = 8000;

    opt.template_path.clear();
    opt.template_neighbor = false;
    opt.template_neighbor_min = 1;
    opt.inspect = true;

    opt.singledie_normalize.background.method = singledie::BackgroundMethod::Gaussian;
    opt.singledie_normalize.background.kernelSize = 201;
    opt.singledie_normalize.background.sigma = 0.0;
    opt.singledie_normalize.method = singledie::IntensityMethod::Divide;
    opt.singledie_normalize.minBackground = 1e-6f;
    opt.singledie_normalize.robustPercentile = true;

    // 本片只走 largeWhite 抓贴边小白点；关掉残差亮/暗，避免把亮环当缺陷。
    opt.singledie_detect.enableBright = false;
    opt.singledie_detect.enableDark = false;
    opt.singledie_detect.brightThreshold = 0.025f;
    opt.singledie_detect.darkThreshold = 0.025f;
    opt.singledie_detect.morphologyKernel = 3;
    opt.singledie_detect.openIterations = 0;
    opt.singledie_detect.closeIterations = 1;
    opt.singledie_detect.minArea = 1;
    opt.singledie_detect.maxArea = 10000000;
    opt.singledie_detect.borderMargin = 0;
    opt.singledie_detect.enableBlob = false;
    opt.singledie_detect.keepResponseMaps = false;
    opt.singledie_detect.maxAspectRatio = 0.0f;
    opt.singledie_detect.minFillRatio = 0.0f;

    opt.eval_patch_dir.clear();
    opt.eval_truth_dir.clear();

    apply_common_backend(opt);

    opt.defect.adaptive.blockSize = 31;
    opt.defect.adaptive.k = 4.0;
    opt.defect.adaptive.minAbs = 18.0;

    opt.defect.signedDiff.T = 16.0;
    opt.defect.signedDiff.darkT = 14.0;

    opt.defect.darkSpot.enableBlackHat = true;
    opt.defect.darkSpot.kernel = 15;
    opt.defect.darkSpot.minResponse = 12.0;

    opt.defect.morph.openSize = 0;
    opt.defect.morph.closeSize = 3;
    opt.defect.morph.dilateSize = 0;
    opt.defect.morph.erodeSize = 0;
    opt.defect.morph.removeNoiseSize = 0;
    opt.defect.morph.darkOpenSize = 0;

    opt.defect.blob.minArea = 8;
    opt.defect.blob.maxArea = 250;
    opt.defect.blob.minWidth = 3;
    opt.defect.blob.minHeight = 3;
    opt.defect.blob.maxAspectRatio = 4.0;
    opt.defect.blob.borderMargin = 4;
    opt.defect.blob.borderStripAspect = 2.5;
    opt.defect.blob.minBrightArea = 400;

    opt.defect.filter.keepBright = false;
    opt.defect.filter.keepDark = false;
    opt.defect.filter.keepMixed = false;
    opt.defect.filter.keepUnknown = false;
    opt.defect.filter.minMeanL2 = 14.0;
    opt.defect.filter.minAbsMeanSigned = 12.0;

    // 0804-slot3 暗场小白点：左 3 / 右 4 / 下 10，共计 17。
    // 阈值 50 抓住左侧浅点（峰值约 60）；maxArea 丢掉底部 notch（~900px）。
    // 右侧 2 个真点贴边且面积<=4，不能开 isolatedInnerMargin。
    opt.defect.largeWhite.enable = true;
    opt.defect.largeWhite.threshold = 50.0;
    opt.defect.largeWhite.minArea = 2;
    opt.defect.largeWhite.maxArea = 50;
    opt.defect.largeWhite.minFill = 0.50;
    opt.defect.largeWhite.maxAspectRatio = 2.0;
    opt.defect.largeWhite.morphSize = 0;
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
    opt.defect.largeWhite.isolatedMaxArea = 0;
    opt.defect.largeWhite.isolatedThreshold = 0.0;
    opt.defect.largeWhite.isolatedInnerMargin = 0;

    opt.defect.darkRegion.enable = false;
    opt.defect.darkRegion.blackhatSize = 15;
    opt.defect.darkRegion.blackhatMin = 16.0;
    opt.defect.darkRegion.belowMedian = 16.0;
    opt.defect.darkRegion.grayMin = 90.0;
    opt.defect.darkRegion.grayMax = 220.0;
    opt.defect.darkRegion.minArea = 16;
    opt.defect.darkRegion.maxArea = 180;
    opt.defect.darkRegion.minFill = 0.40;
    opt.defect.darkRegion.maxAspectRatio = 2.2;
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
}

}  // namespace

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try {
        // ========== 切换检出模式 ==========
        // DieToDie : 切 die patch，相邻 die 比较（c3000 有图形晶圆）
        // SingleDie: 切圆形 ROI，本片亮/暗检出（c3000-nopartern-wafer 无图形晶圆）
        constexpr cutdie::InspectMode kInspectMode = cutdie::InspectMode::SingleDie;

        cutdie::Options opt;
        if (kInspectMode == cutdie::InspectMode::SingleDie) {
            apply_singledie_recipe(opt);
        } else {
            apply_d2d_recipe(opt);
        }

        if (opt.resolution_um <= 0.0) {
            throw std::runtime_error("每个像素对应的 um 必须大于 0");
        }
        return cutdie::run(opt);
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << '\n';
        return 1;
    }
}
