#include "die_config.h"

namespace cutdie {

Options make_default_options() {
    Options opt;

    // 输入 / 输出
    opt.dir = ".";
    opt.image.clear();
    opt.xml.clear();
    opt.out.clear();
    opt.patch_out.clear();
    opt.resolution_um = 46.0;
    opt.keep_skipped = false;
    opt.recursive = false;
    opt.overlay_preview_px = 8000;

    // 模板 / 检出
    opt.template_path.clear();
    opt.template_neighbor = true;
    opt.template_neighbor_min = 2;
    opt.inspect = true;
    opt.eval_patch_dir.clear();
    opt.eval_truth_dir.clear();

    // 后端
    opt.align_method = AlignMethod::PhaseCpu;
    opt.histogram_method = HistogramMethod::Cpu;
    opt.normalize_method = NormalizeMethod::Cpu;

    // 直方图阈值
    opt.histogram.gaussianSigma = 2.5f;
    opt.histogram.peakHeightRatio = 0.05f;
    opt.histogram.minPeakDistance = 15;
    opt.histogram.minPeakCountRatio = 0.005f;
    opt.histogram.defaultThreshold = 125;

    // ECC 对位
    opt.ecc.maxIters = 8;
    opt.ecc.eps = 1e-5;
    opt.ecc.gaussFiltSize = 3;

    // 缺陷检测配方
    opt.defect.adaptive.blockSize = 31;
    opt.defect.adaptive.k = 4.0;
    opt.defect.adaptive.minAbs = 14.0;
    opt.defect.signedDiff.T = 14.0;
    opt.defect.signedDiff.darkT = 10.0;
    opt.defect.darkSpot.enableBlackHat = false;
    opt.defect.darkSpot.kernel = 15;
    opt.defect.darkSpot.minResponse = 8.0;
    opt.defect.morph.openSize = 3;
    opt.defect.morph.closeSize = 3;
    opt.defect.morph.dilateSize = 0;
    opt.defect.morph.erodeSize = 0;
    opt.defect.morph.removeNoiseSize = 0;
    opt.defect.morph.darkOpenSize = 0;
    opt.defect.blob.minArea = 25;
    opt.defect.blob.maxArea = 0;
    opt.defect.blob.minWidth = 4;
    opt.defect.blob.minHeight = 4;
    opt.defect.blob.maxAspectRatio = 3.0;
    opt.defect.blob.borderMargin = 4;
    opt.defect.blob.borderStripAspect = 2.5;
    opt.defect.blob.minBrightArea = 400;
    opt.defect.filter.keepBright = true;
    opt.defect.filter.keepDark = true;
    opt.defect.filter.keepMixed = false;
    opt.defect.filter.keepUnknown = false;
    opt.defect.filter.minMeanL2 = 10.0;
    opt.defect.filter.minAbsMeanSigned = 8.0;
    opt.defect.largeWhite.enable = true;
    opt.defect.largeWhite.threshold = 230.0;
    opt.defect.largeWhite.minArea = 4000;
    opt.defect.largeWhite.minFill = 0.15;
    opt.defect.largeWhite.maxAspectRatio = 3.0;
    return opt;
}

}  // namespace cutdie
