// 同一工程内切换 Single Die / Die-to-Die / Die-to-Database / yolo-openvino。
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

// 无图形晶圆：按 XML 切并保存全部 die patch，再把圆形 ROI 送入 SingleDie。
void apply_singledie_recipe(cutdie::Options& opt) {
    opt.mode = cutdie::InspectMode::SingleDie;

    opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-20261009";
    opt.image = "C:/Users/aspiring135/Desktop/datasets/50um-20261009/0907_PW_slot2.png";
    opt.xml = "C:/Users/aspiring135/Desktop/datasets/50um-20261009/0907_PW_slot2.xml";
    opt.out = "C:/Users/aspiring135/Desktop/datasets/50um-20261009";
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

    // 阈值 70。minArea=1 保留暗底上只有中心超过阈值的小白点。
    // maxArea=3000、maxAspectRatio=50：收下顶部整段亮带。
    // minFill=0.33：斜向小白点串填充率约 0.33～0.53，不能再用 0.55 丢掉。
    // inner_m=0：贴边 1～4px 的小白点也保留。
    opt.defect.largeWhite.enable = true;
    opt.defect.largeWhite.threshold = 70.0;
    opt.defect.largeWhite.minArea = 1;
    opt.defect.largeWhite.maxArea = 3000;
    opt.defect.largeWhite.minFill = 0.33;
    opt.defect.largeWhite.maxAspectRatio = 50.0;
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
    opt.defect.largeWhite.isolatedNearKeep = 0;
    opt.defect.largeWhite.isolatedFarMargin = 0;

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

// 有图形晶圆：切割 die patch，用相邻 die 做 D2D。
void apply_d2d_recipe(cutdie::Options& opt) {
    opt.mode = cutdie::InspectMode::DieToDie;

    opt.dir = "C:/Users/aspiring135/Desktop/datasets/46um-gray";
    opt.image = "C:/Users/aspiring135/Desktop/datasets/46um-gray/0046.png";
    opt.xml = "C:/Users/aspiring135/Desktop/datasets/46um-gray/0046.xml";
    opt.out = "C:/Users/aspiring135/Desktop/datasets/46um-gray/0046-DieToDie";
    opt.patch_out.clear();
    opt.resolution_um = 46.0;
    opt.keep_skipped = false;
    opt.skip_inspect.clear();
    // 只检出这些晶粒上的缺陷。
    opt.inspect_only = {
        {-10, -6}, {-11, -3}, {-11, -4}, {-14, 1}, {-14, 2}, {-14, 4},
        {-1, -5}, {-2, -13}, {-2, -4}, {-2, 2}, {-5, 12}, {0, -14}, {0, -7},
        {10, 7}, {12, 2}, {13, -3}, {13, -4}, {13, 4}, {1, -14}, {1, 11}, {1, 4},
        {2, -11}, {3, -3}, {3, -7}, {4, -1}, {5, -11}, {5, 5}, {6, 6}, {7, 11},
        {8, -8}, {9, 2},
    };
    opt.recursive = false;
    opt.overlay_preview_px = 8000;

    opt.template_path.clear();
    opt.template_neighbor = true;
    opt.template_neighbor_min = 1;
    opt.inspect = true;

    opt.eval_patch_dir.clear();
    opt.eval_truth_dir.clear();

    apply_common_backend(opt);

    opt.defect.adaptive.blockSize = 31;
    opt.defect.adaptive.k = 2.5;
    opt.defect.adaptive.minAbs = 12.0;

    opt.defect.signedDiff.T = 14.0;
    opt.defect.signedDiff.darkT = 10.0;

    opt.defect.darkSpot.enableBlackHat = true;
    opt.defect.darkSpot.kernel = 15;
    opt.defect.darkSpot.minResponse = 12.0;

    opt.defect.morph.openSize = 0;
    opt.defect.morph.closeSize = 0;
    opt.defect.morph.dilateSize = 0;
    opt.defect.morph.erodeSize = 0;
    opt.defect.morph.removeNoiseSize = 0;
    opt.defect.morph.darkOpenSize = 0;

    opt.defect.blob.minArea = 8;
    opt.defect.blob.maxArea = 800;
    opt.defect.blob.minWidth = 3;
    opt.defect.blob.minHeight = 3;
    opt.defect.blob.maxAspectRatio = 3.0;
    opt.defect.blob.borderMargin = 2;
    opt.defect.blob.borderStripAspect = 4.0;
    opt.defect.blob.minBrightArea = 12;

    // 这批晶粒里既有浅黑点，也有小白点。大白块仍走 largeWhite。
    opt.defect.filter.keepBright = true;
    opt.defect.filter.keepDark = true;
    opt.defect.filter.keepMixed = false;
    opt.defect.filter.keepUnknown = false;
    opt.defect.filter.minMeanL2 = 14.0;
    opt.defect.filter.minAbsMeanSigned = 12.0;

    // 中心正方形约 107x105 到 131x105，长宽比约 1.25，面积约 10800。
    // 贴边白条约 6–7。长宽比上限 2 只留正方形。
    opt.defect.largeWhite.enable = true;
    opt.defect.largeWhite.threshold = 236.0;
    opt.defect.largeWhite.minArea = 1000;
    opt.defect.largeWhite.maxArea = 20000;
    opt.defect.largeWhite.minFill = 0.35;
    opt.defect.largeWhite.maxAspectRatio = 2.0;
    opt.defect.largeWhite.morphSize = 3;
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

// 切 die patch 后，用一张参考 patch 和每颗 patch 对比。暗斑、亮斑走差分，大白块另走 largeWhite。
void apply_die_to_database_recipe(cutdie::Options& opt) {
    opt.mode = cutdie::InspectMode::DieToDatabase;
    opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-gray";
    opt.image = "C:/Users/aspiring135/Desktop/datasets/50um-gray/0050.png";
    opt.xml = "C:/Users/aspiring135/Desktop/datasets/50um-gray/0050.xml";
    opt.out = "C:/Users/aspiring135/Desktop/datasets/50um-gray/0050-DieToDatabase";
    opt.patch_out.clear();
    opt.resolution_um = 50.0;
    opt.keep_skipped = false;
    // 四个角、x-14_y-2，以及这些过检暗点，不检出。
    opt.skip_inspect = {{6, 12}, {6, -12}, {-7, 12}, {-7, -12}, {-14, -2},
                        {1, 7}, {7, 6}, {7, 7}, {-10, 2},
                        {-3, 1}, {-3, 4}, {-3, 12}, {3, 13}, {10, 6}, {-11, 4}};
    opt.recursive = false;
    opt.overlay_preview_px = 8000;

    // 单张参考 patch。放在 patches 目录外，切图清 png 时不会被删掉。
    opt.template_path = "C:/Users/aspiring135/Desktop/datasets/50um-gray/reference.png";
    opt.template_neighbor = false;
    opt.template_neighbor_min = 1;
    opt.inspect = true;

    opt.eval_patch_dir.clear();
    opt.eval_truth_dir.clear();

    apply_common_backend(opt);

    // 参考 die 和远处晶粒有十几灰度的照度差。k 太大、black-hat 太松时，
    // 残差会把自适应阈值抬高，真暗斑被打碎，同时留下上万个碎点。
    opt.defect.adaptive.blockSize = 31;
    opt.defect.adaptive.k = 2.5;
    opt.defect.adaptive.minAbs = 16.0;

    opt.defect.signedDiff.T = 16.0;
    opt.defect.signedDiff.darkT = 16.0;

    opt.defect.darkSpot.enableBlackHat = false;
    opt.defect.darkSpot.kernel = 15;
    opt.defect.darkSpot.minResponse = 16.0;

    // 闭运算会把切割道上的浅斑连成长条，再被长宽比丢掉。
    opt.defect.morph.openSize = 0;
    opt.defect.morph.closeSize = 0;
    opt.defect.morph.dilateSize = 0;
    opt.defect.morph.erodeSize = 0;
    opt.defect.morph.removeNoiseSize = 0;
    opt.defect.morph.darkOpenSize = 0;

    opt.defect.blob.minArea = 17;
    opt.defect.blob.maxArea = 2000;
    opt.defect.blob.minWidth = 4;
    opt.defect.blob.minHeight = 4;
    opt.defect.blob.maxAspectRatio = 2.6;
    opt.defect.blob.borderMargin = 4;
    opt.defect.blob.borderStripAspect = 3.0;
    opt.defect.blob.minBrightArea = 0;

    // 其它晶粒上的黑色小点仍走差分。有大白块的晶粒在检出时丢掉旁边的假暗点。
    opt.defect.filter.keepBright = false;
    opt.defect.filter.keepDark = true;
    opt.defect.filter.keepMixed = false;
    opt.defect.filter.keepUnknown = false;
    opt.defect.filter.minMeanL2 = 16.0;
    opt.defect.filter.minAbsMeanSigned = 14.0;

    // 晶粒底色大约 150。阈值必须高于底色，否则整颗 die 会连成一块。
    // 白色正方形约 107x105 到 131x105，长宽比约 1.25，面积约 10800。
    // 贴边白条约 6–7。长宽比上限 2 只留正方形。
    opt.defect.largeWhite.enable = true;
    opt.defect.largeWhite.threshold = 236.0;
    opt.defect.largeWhite.minArea = 1000;
    opt.defect.largeWhite.maxArea = 20000;
    opt.defect.largeWhite.minFill = 0.35;
    opt.defect.largeWhite.maxAspectRatio = 2.0;
    opt.defect.largeWhite.morphSize = 3;
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

    // 暗斑已经走模板差分。这里再开一次会和差分结果重复。
    opt.defect.darkRegion.enable = false;
}

// 有图形晶圆：切割 die patch，用 OpenVINO YOLO26 语义分割推理。
void apply_yolo_openvino_recipe(cutdie::Options& opt) {
    opt.mode = cutdie::InspectMode::YoloOpenVino;

    opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-0804";
    opt.image = "C:/Users/aspiring135/Desktop/datasets/50um-0804/0804-slot1.png";
    opt.xml = "C:/Users/aspiring135/Desktop/datasets/50um-0804/0804-slot1.xml";
    opt.out.clear();
    opt.patch_out.clear();
    opt.resolution_um = 50.0;
    opt.keep_skipped = false;
    opt.recursive = false;
    opt.overlay_preview_px = 8000;
    opt.inspect = true;

    opt.yolo_model = "C:/Users/aspiring135/Desktop/project/c3000/models/best.xml";
    opt.yolo_device = "CPU";
    opt.yolo_imgsz = 64;
    opt.yolo_batch = 8;
    opt.yolo_requests = 0;
    opt.yolo_confidence = 0.25f;
    opt.yolo_fp16 = true;
    opt.yolo_throughput = false;

    opt.eval_patch_dir.clear();
    opt.eval_truth_dir.clear();
}

}  // namespace

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try {
        // ========== 切换检出模式 ==========
        // SingleDie     : 切并保存全部 die patch，再做圆形 ROI 亮/暗检出
        // DieToDie      : 切 die patch，相邻 die 比较
        // DieToDatabase : 切 die patch，用一张参考 patch 模板做同样的对比
        // YoloOpenVino  : 切 die patch，OpenVINO YOLO26 语义分割（best.xml，64x64）
        constexpr cutdie::InspectMode kInspectMode = cutdie::InspectMode::DieToDie;

        cutdie::Options opt;
        if (kInspectMode == cutdie::InspectMode::SingleDie) {
            apply_singledie_recipe(opt);
        } else if (kInspectMode == cutdie::InspectMode::DieToDie) {
            apply_d2d_recipe(opt);
        } else if (kInspectMode == cutdie::InspectMode::DieToDatabase) {
            apply_die_to_database_recipe(opt);
        } else if (kInspectMode == cutdie::InspectMode::YoloOpenVino) {
            apply_yolo_openvino_recipe(opt);
        }else if (kInspectMode == cutdie::InspectMode::AnomalibOpenVino) {
        
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
