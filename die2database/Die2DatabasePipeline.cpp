#include "Die2DatabasePipeline.h"

#include "EccAlign.cuh"
#include "EccAlign.hpp"
#include "HistogramThreshold.cuh"
#include "IntensityNormalize.cuh"
#include "IntensityNormalize.h"
#include "PhaseCorrelate.cuh"
#include "PhaseCorrelate.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace cutdie {
namespace {

int oddSize(int size, int minimum) {
    if (size < minimum) {
        return 0;
    }
    return (size % 2) == 0 ? size + 1 : size;
}

cv::Mat asContinuous(const cv::Mat& src) {
    return src.isContinuous() ? src : src.clone();
}

cv::Mat to8U(const cv::Mat& src) {
    if (src.depth() == CV_8U) {
        return src;
    }
    cv::Mat dst;
    if (src.depth() == CV_16U) {
        src.convertTo(dst, CV_8U, 1.0 / 256.0);
        return dst;
    }
    double maxv = 0.0;
    cv::minMaxLoc(src, nullptr, &maxv);
    src.convertTo(dst, CV_8U, maxv <= 1.5 ? 255.0 : 1.0);
    return dst;
}

cv::Mat toGray(const cv::Mat& src) {
    const cv::Mat image = to8U(src);
    if (image.channels() == 1) {
        return asContinuous(image);
    }
    cv::Mat gray;
    if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    return asContinuous(gray);
}

cv::Mat toBgr(const cv::Mat& src) {
    if (src.empty()) {
        return {};
    }
    if (src.channels() == 3) {
        return src.clone();
    }
    if (src.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

DefectPolarity classifyPolarity(int brightArea, int darkArea) {
    if (brightArea <= 0 && darkArea <= 0) {
        return DefectPolarity::Unknown;
    }
    if (brightArea >= darkArea * 2 && brightArea > 0) {
        return DefectPolarity::Bright;
    }
    if (darkArea >= brightArea * 2 && darkArea > 0) {
        return DefectPolarity::Dark;
    }
    return DefectPolarity::Mixed;
}

bool keepPolarity(DefectPolarity polarity, const DefectFilterParams& filter) {
    switch (polarity) {
        case DefectPolarity::Bright:
            return filter.keepBright;
        case DefectPolarity::Dark:
            return filter.keepDark;
        case DefectPolarity::Mixed:
            return filter.keepMixed;
        default:
            return filter.keepUnknown;
    }
}

void applyMorph(cv::Mat& mask, int size, cv::MorphTypes op) {
    const int k = oddSize(size, 3);
    if (k < 3 || mask.empty()) {
        return;
    }
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
    cv::morphologyEx(mask, mask, op, kernel);
}

double medianOf(std::vector<float> values) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t mid = (values.size() - 1) / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    return values[mid];
}

cv::Mat adaptiveMask(const cv::Mat& fused, const AdaptiveThresholdParams& params) {
    int block = params.blockSize > 0 ? params.blockSize : std::max(15, std::min(fused.cols, fused.rows) / 6);
    block = oddSize(block, 3);
    cv::Mat mean;
    cv::Mat meanSq;
    cv::Mat sq;
    cv::GaussianBlur(fused, mean, cv::Size(block, block), 0.0);
    cv::multiply(fused, fused, sq);
    cv::GaussianBlur(sq, meanSq, cv::Size(block, block), 0.0);
    cv::Mat var = meanSq - mean.mul(mean);
    cv::max(var, 0.0, var);
    cv::Mat localStd;
    cv::sqrt(var, localStd);
    cv::Mat thresh = mean + params.k * localStd;
    cv::max(thresh, params.minAbs, thresh);
    cv::Mat local;
    cv::compare(fused, thresh, local, cv::CMP_GT);

    const int n = fused.rows * fused.cols;
    std::vector<float> values(static_cast<size_t>(n));
    if (fused.isContinuous()) {
        std::copy(fused.ptr<float>(), fused.ptr<float>() + n, values.begin());
    } else {
        int i = 0;
        for (int y = 0; y < fused.rows; ++y) {
            const float* row = fused.ptr<float>(y);
            for (int x = 0; x < fused.cols; ++x) {
                values[static_cast<size_t>(i++)] = row[x];
            }
        }
    }
    const double median = medianOf(values);
    for (float& value : values) {
        value = std::abs(value - static_cast<float>(median));
    }
    const double mad = medianOf(values);
    const double globalT = mad < 1e-9 ? params.minAbs : std::max(params.minAbs, median + params.k * 1.4826 * mad);
    cv::Mat global;
    cv::compare(fused, globalT, global, cv::CMP_GT);
    cv::Mat mask;
    cv::bitwise_or(local, global, mask);
    return mask;
}

struct DiffDefects {
    std::vector<DefectCandidate> candidates;
    cv::Mat blobMask;
};

DiffDefects detectDifference(const cv::Mat& templateNormalized, const cv::Mat& dataNormalized,
                             const cv::Mat& l2Difference, const DefectDetectParams& params) {
    DiffDefects out;
    const cv::Mat templ = toGray(templateNormalized);
    const cv::Mat data = toGray(dataNormalized);
    if (templ.size() != data.size() || l2Difference.empty()) {
        return out;
    }
    cv::Mat templF;
    cv::Mat dataF;
    templ.convertTo(templF, CV_32F);
    data.convertTo(dataF, CV_32F);
    const cv::Mat signedDiff = dataF - templF;
    cv::Mat l2;
    if (l2Difference.type() == CV_32FC1 && l2Difference.size() == signedDiff.size()) {
        l2 = l2Difference;
    } else {
        cv::absdiff(templF, dataF, l2);
    }

    const double brightT = params.signedDiff.T > 0.0 ? params.signedDiff.T : std::max(1.0, params.adaptive.minAbs);
    const double darkT = params.signedDiff.darkT > 0.0 ? params.signedDiff.darkT : brightT;
    cv::Mat brightMask;
    cv::Mat darkMask;
    cv::compare(signedDiff, brightT, brightMask, cv::CMP_GT);
    cv::compare(signedDiff, -darkT, darkMask, cv::CMP_LT);
    if (params.darkSpot.enableBlackHat) {
        const int k = oddSize(params.darkSpot.kernel, 3);
        if (k >= 3) {
            const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
            cv::Mat hat;
            cv::morphologyEx(data, hat, cv::MORPH_BLACKHAT, kernel);
            cv::Mat hatMask;
            cv::compare(hat, params.darkSpot.minResponse, hatMask, cv::CMP_GT);
            cv::Mat signedDark;
            cv::compare(signedDiff, -std::max(1.0, darkT * 0.5), signedDark, cv::CMP_LT);
            cv::bitwise_and(hatMask, signedDark, hatMask);
            cv::bitwise_or(darkMask, hatMask, darkMask);
        }
    }
    cv::Mat polarity;
    cv::bitwise_or(brightMask, darkMask, polarity);
    cv::Mat fused = l2.clone();
    fused.setTo(0, polarity == 0);

    cv::Mat mask = adaptiveMask(fused, params.adaptive);
    applyMorph(mask, params.morph.openSize, cv::MORPH_OPEN);
    applyMorph(mask, params.morph.closeSize, cv::MORPH_CLOSE);
    applyMorph(mask, params.morph.dilateSize, cv::MORPH_DILATE);
    applyMorph(mask, params.morph.erodeSize, cv::MORPH_ERODE);
    applyMorph(mask, params.morph.removeNoiseSize, cv::MORPH_OPEN);
    if (oddSize(params.morph.darkOpenSize, 3) >= 3) {
        cv::Mat dark = darkMask.clone();
        applyMorph(dark, params.morph.darkOpenSize, cv::MORPH_OPEN);
        cv::bitwise_or(mask, dark, mask);
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    for (int i = 1; i < count; ++i) {
        DefectCandidate candidate;
        candidate.labelId = i;
        candidate.area = stats.at<int>(i, cv::CC_STAT_AREA);
        candidate.bbox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                                  stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        candidate.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        double l2Sum = 0.0;
        double l2Max = 0.0;
        double signedSum = 0.0;
        int n = 0;
        const cv::Rect roi = candidate.bbox & cv::Rect(0, 0, mask.cols, mask.rows);
        for (int y = roi.y; y < roi.y + roi.height; ++y) {
            const int* lab = labels.ptr<int>(y);
            const float* l2Row = l2.ptr<float>(y);
            const float* signedRow = signedDiff.ptr<float>(y);
            const unsigned char* brightRow = brightMask.ptr<unsigned char>(y);
            const unsigned char* darkRow = darkMask.ptr<unsigned char>(y);
            for (int x = roi.x; x < roi.x + roi.width; ++x) {
                if (lab[x] != i) {
                    continue;
                }
                l2Sum += l2Row[x];
                l2Max = std::max(l2Max, static_cast<double>(l2Row[x]));
                signedSum += signedRow[x];
                if (brightRow[x]) {
                    ++candidate.brightArea;
                }
                if (darkRow[x]) {
                    ++candidate.darkArea;
                }
                ++n;
            }
        }
        if (n > 0) {
            candidate.meanL2 = l2Sum / static_cast<double>(n);
            candidate.maxL2 = l2Max;
            candidate.meanSigned = signedSum / static_cast<double>(n);
        }
        candidate.polarity = classifyPolarity(candidate.brightArea, candidate.darkArea);
        int minArea = params.blob.minArea;
        if (candidate.polarity == DefectPolarity::Bright && params.blob.minBrightArea > 0) {
            minArea = std::max(minArea, params.blob.minBrightArea);
        }
        if (candidate.area < minArea || (params.blob.maxArea > 0 && candidate.area > params.blob.maxArea)) {
            continue;
        }
        if (candidate.bbox.width < params.blob.minWidth || candidate.bbox.height < params.blob.minHeight) {
            continue;
        }
        const int shortSide = std::min(candidate.bbox.width, candidate.bbox.height);
        const double aspect =
            shortSide > 0 ? static_cast<double>(std::max(candidate.bbox.width, candidate.bbox.height)) /
                                static_cast<double>(shortSide)
                          : 0.0;
        if (aspect > params.blob.maxAspectRatio) {
            continue;
        }
        if (params.blob.borderMargin > 0) {
            const int m = params.blob.borderMargin;
            const bool touches = candidate.bbox.x <= m || candidate.bbox.y <= m ||
                                 candidate.bbox.x + candidate.bbox.width >= mask.cols - m ||
                                 candidate.bbox.y + candidate.bbox.height >= mask.rows - m;
            if (touches && aspect > params.blob.borderStripAspect) {
                continue;
            }
        }
        if (!keepPolarity(candidate.polarity, params.filter) || candidate.meanL2 < params.filter.minMeanL2 ||
            std::abs(candidate.meanSigned) < params.filter.minAbsMeanSigned) {
            continue;
        }
        out.candidates.push_back(candidate);
    }

    out.blobMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int y = 0; y < labels.rows; ++y) {
        const int* lab = labels.ptr<int>(y);
        unsigned char* dst = out.blobMask.ptr<unsigned char>(y);
        for (int x = 0; x < labels.cols; ++x) {
            for (const auto& candidate : out.candidates) {
                if (candidate.labelId == lab[x]) {
                    dst[x] = 255;
                    break;
                }
            }
        }
    }
    return out;
}

std::vector<DefectCandidate> componentsOf(const cv::Mat& mask, const cv::Mat& gray, int minArea, int maxArea,
                                          double minFill, double maxAspect, DefectPolarity polarity) {
    std::vector<DefectCandidate> found;
    if (mask.empty()) {
        return found;
    }
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    for (int i = 1; i < count; ++i) {
        DefectCandidate candidate;
        candidate.area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (candidate.area < minArea || (maxArea > 0 && candidate.area > maxArea)) {
            continue;
        }
        candidate.bbox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                                  stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        const int shortSide = std::min(candidate.bbox.width, candidate.bbox.height);
        const double aspect =
            shortSide > 0 ? static_cast<double>(std::max(candidate.bbox.width, candidate.bbox.height)) /
                                static_cast<double>(shortSide)
                          : 0.0;
        const double boxArea = static_cast<double>(candidate.bbox.area());
        if (aspect > maxAspect || boxArea <= 0.0 || static_cast<double>(candidate.area) / boxArea < minFill) {
            continue;
        }
        candidate.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        candidate.polarity = polarity;
        double sum = 0.0;
        double peak = 0.0;
        int n = 0;
        for (int y = candidate.bbox.y; y < candidate.bbox.y + candidate.bbox.height; ++y) {
            const int* lab = labels.ptr<int>(y);
            const unsigned char* row = gray.ptr<unsigned char>(y);
            for (int x = candidate.bbox.x; x < candidate.bbox.x + candidate.bbox.width; ++x) {
                if (lab[x] != i) {
                    continue;
                }
                sum += row[x];
                peak = std::max(peak, static_cast<double>(row[x]));
                ++n;
            }
        }
        if (n > 0) {
            candidate.meanL2 = sum / static_cast<double>(n);
            candidate.maxL2 = peak;
            candidate.meanSigned = polarity == DefectPolarity::Dark ? -candidate.meanL2 : candidate.meanL2;
        }
        if (polarity == DefectPolarity::Bright) {
            candidate.brightArea = candidate.area;
        } else {
            candidate.darkArea = candidate.area;
        }
        found.push_back(candidate);
    }
    return found;
}

}  // namespace

struct DatabaseComparator::Impl {
    Options opt;
    HistogramThreshold histogram;
    PhaseCorrelate phase;
    std::unique_ptr<PhaseCorrelateCuda> phaseCuda;
    std::unique_ptr<EccAlignCuda> eccCuda;
    std::unique_ptr<IntensityNormalizeCuda> intensityCuda;
    bool cudaAlign = false;
    bool cudaHist = false;
    bool cudaNorm = false;

    explicit Impl(const Options& options) : opt(options), histogram(options.histogram) {
        cudaAlign = opt.align_method == AlignMethod::PhaseCuda || opt.align_method == AlignMethod::EccCuda;
        cudaHist = opt.histogram_method == HistogramMethod::Cuda ||
                   (opt.histogram_method == HistogramMethod::Auto && cudaAlign);
        cudaNorm = opt.normalize_method == NormalizeMethod::Cuda ||
                   (opt.normalize_method == NormalizeMethod::Auto && cudaAlign);
        if ((cudaAlign || cudaHist || cudaNorm) && cudaSetDevice(0) != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice 失败");
        }
        if (cudaAlign) {
            phaseCuda = std::make_unique<PhaseCorrelateCuda>();
            eccCuda = std::make_unique<EccAlignCuda>();
            EccCudaParams cudaEcc{};
            cudaEcc.maxIters = opt.ecc.maxIters;
            cudaEcc.eps = opt.ecc.eps;
            cudaEcc.gaussFiltSize = opt.ecc.gaussFiltSize;
            cudaEcc.motionType = opt.ecc.motionType;
            eccCuda->setParams(cudaEcc);
        }
        if (cudaNorm) {
            intensityCuda = std::make_unique<IntensityNormalizeCuda>();
        }
    }
};

DatabaseComparator::DatabaseComparator(const Options& opt) : impl_(std::make_unique<Impl>(opt)) {}

DatabaseComparator::~DatabaseComparator() = default;

std::string DatabaseComparator::backendText() const {
    const char* align = "phase-cpu";
    if (impl_->opt.align_method == AlignMethod::PhaseCuda) {
        align = "phase-cuda";
    } else if (impl_->opt.align_method == AlignMethod::EccCpu) {
        align = "ecc-cpu";
    } else if (impl_->opt.align_method == AlignMethod::EccCuda) {
        align = "ecc-cuda";
    }
    return std::string("align=") + align + "  histogram=" + (impl_->cudaHist ? "cuda" : "cpu") +
           "  normalize=" + (impl_->cudaNorm ? "cuda" : "cpu");
}

DatabaseMatch DatabaseComparator::match(const cv::Mat& reference, const cv::Mat& patch) {
    DatabaseMatch result;
    cv::Mat templateOrig = to8U(reference);
    cv::Mat dataOrig = to8U(patch);
    if (templateOrig.channels() != dataOrig.channels()) {
        templateOrig = toBgr(templateOrig);
        dataOrig = toBgr(dataOrig);
    }
    templateOrig = asContinuous(templateOrig);
    dataOrig = asContinuous(dataOrig);
    const cv::Mat templateGray = toGray(templateOrig);
    const cv::Mat dataGray = toGray(dataOrig);
    if (templateGray.size() != dataGray.size()) {
        result.error = "模板和 patch 尺寸不一致";
        return result;
    }

    const int w = templateGray.cols;
    const int h = templateGray.rows;
    const int n = w * h;
    cv::Mat templateBin(h, w, CV_8UC1);
    cv::Mat dataBin(h, w, CV_8UC1);
    HistogramThresholdResult templateHist{};
    HistogramThresholdResult dataHist{};
    if (impl_->cudaHist) {
        if (impl_->histogram.computeFromGray(templateGray.data, w, h, templateHist) != cudaSuccess ||
            impl_->histogram.computeFromGray(dataGray.data, w, h, dataHist) != cudaSuccess ||
            impl_->histogram.apply(templateGray.data, n, templateHist.threshold, templateBin.data) != cudaSuccess ||
            impl_->histogram.apply(dataGray.data, n, dataHist.threshold, dataBin.data) != cudaSuccess) {
            result.error = "直方图二值化失败";
            return result;
        }
    } else if (!impl_->histogram.computeFromGrayCpu(templateGray.data, w, h, templateHist) ||
               !impl_->histogram.computeFromGrayCpu(dataGray.data, w, h, dataHist) ||
               !impl_->histogram.applyCpu(templateGray.data, n, templateHist.threshold, templateBin.data) ||
               !impl_->histogram.applyCpu(dataGray.data, n, dataHist.threshold, dataBin.data)) {
        result.error = "直方图二值化失败";
        return result;
    }

    cv::Point2d shift;
    const bool ecc = impl_->opt.align_method == AlignMethod::EccCpu || impl_->opt.align_method == AlignMethod::EccCuda;
    if (ecc) {
        if (impl_->cudaAlign) {
            EccCudaResult eccResult{};
            if (impl_->eccCuda->compute(templateGray.data, dataGray.data, w, h, eccResult) != cudaSuccess ||
                !eccResult.ok) {
                result.error = "ECC 对位没有收敛";
                return result;
            }
            shift = cv::Point2d(eccResult.dx, eccResult.dy);
        } else {
            const EccAlignResult eccResult = eccShift(templateGray, dataGray, impl_->opt.ecc);
            if (!eccResult.ok) {
                result.error = eccResult.error.empty() ? "ECC 对位没有收敛" : eccResult.error;
                return result;
            }
            shift = eccResult.shift;
        }
    } else if (impl_->cudaAlign) {
        PhaseCorrelateResult phase{};
        if (impl_->phaseCuda->compute(templateBin.data, dataBin.data, w, h, phase) != cudaSuccess) {
            result.error = "相位相关对位失败";
            return result;
        }
        shift = cv::Point2d(phase.dx, phase.dy);
    } else {
        shift = impl_->phase.compute(templateBin, dataBin, nullptr);
    }

    const cv::Mat dataAligned = shiftImage(dataOrig, shift, cv::INTER_LINEAR);
    result.commonRoi = commonOverlapRoi(templateOrig.size(), shift);
    if (result.commonRoi.width < 16 || result.commonRoi.height < 16) {
        result.error = "模板和 patch 的重叠区域过小";
        return result;
    }
    const cv::Mat templateCommon = templateOrig(result.commonRoi).clone();
    const cv::Mat dataCommon = dataAligned(result.commonRoi).clone();
    const IntensityNormalizeOutput normalized =
        impl_->cudaNorm ? intensityNormalizeCuda(*impl_->intensityCuda, templateCommon, dataCommon)
                        : intensityNormalize(templateCommon, dataCommon);
    if (!normalized.ok) {
        result.error = normalized.error.empty() ? "亮度归一失败" : normalized.error;
        return result;
    }

    const DiffDefects defects = detectDifference(normalized.templateNormalized, normalized.dataNormalized,
                                                 normalized.l2Difference, impl_->opt.defect);
    result.candidates = defects.candidates;
    result.blobMask = defects.blobMask;
    result.ok = true;
    return result;
}

std::vector<DefectCandidate> DatabaseComparator::findBrightBlocks(const cv::Mat& patch) const {
    const LargeWhiteParams& params = impl_->opt.defect.largeWhite;
    if (!params.enable) {
        return {};
    }
    const cv::Mat gray = toGray(patch);
    cv::Mat mask;
    cv::compare(gray, params.threshold, mask, cv::CMP_GE);
    if (oddSize(params.morphSize, 3) >= 3) {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(oddSize(params.morphSize, 3),
                                                                                     oddSize(params.morphSize, 3)));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    }
    std::vector<DefectCandidate> found =
        componentsOf(mask, gray, params.minArea, params.maxArea, params.minFill, params.maxAspectRatio,
                     DefectPolarity::Bright);
    if (params.tophatSize >= 3 && params.tophatMin > 0.0) {
        const int k = oddSize(params.tophatSize, 3);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::Mat hat;
        cv::morphologyEx(gray, hat, cv::MORPH_TOPHAT, kernel);
        cv::Mat hatMask;
        cv::compare(hat, params.tophatMin, hatMask, cv::CMP_GE);
        cv::Mat brightEnough;
        cv::compare(gray, params.tophatFloor > 0.0 ? params.tophatFloor : 50.0, brightEnough, cv::CMP_GE);
        cv::bitwise_and(hatMask, brightEnough, hatMask);
        const int minArea = params.tophatMinArea > 0 ? params.tophatMinArea : params.minArea;
        const double minFill = params.tophatMinFill > 0.0 ? params.tophatMinFill : params.minFill;
        const double maxAspect = params.tophatMaxAspect > 0.0 ? params.tophatMaxAspect : params.maxAspectRatio;
        auto extra = componentsOf(hatMask, gray, minArea, params.tophatMaxArea, minFill, maxAspect,
                                  DefectPolarity::Bright);
        found.insert(found.end(), extra.begin(), extra.end());
    }
    return found;
}

std::vector<DefectCandidate> DatabaseComparator::findDarkBlocks(const cv::Mat& patch) const {
    const DarkRegionParams& params = impl_->opt.defect.darkRegion;
    if (!params.enable || oddSize(params.blackhatSize, 3) < 3) {
        return {};
    }
    const cv::Mat gray = toGray(patch);
    const int k = oddSize(params.blackhatSize, 3);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
    cv::Mat hat;
    cv::morphologyEx(gray, hat, cv::MORPH_BLACKHAT, kernel);
    cv::Mat mask;
    cv::compare(hat, params.blackhatMin, mask, cv::CMP_GE);
    std::vector<unsigned char> pixels;
    pixels.reserve(static_cast<size_t>(gray.total()));
    for (int y = 0; y < gray.rows; ++y) {
        const unsigned char* row = gray.ptr<unsigned char>(y);
        for (int x = 0; x < gray.cols; ++x) {
            if (row[x] >= params.grayMin && row[x] <= params.grayMax) {
                pixels.push_back(row[x]);
            }
        }
    }
    double gate = 128.0 - params.belowMedian;
    if (!pixels.empty()) {
        const auto mid = pixels.begin() + static_cast<std::ptrdiff_t>(pixels.size() / 2);
        std::nth_element(pixels.begin(), mid, pixels.end());
        gate = static_cast<double>(*mid) - params.belowMedian;
    }
    cv::Mat darkEnough;
    cv::compare(gray, gate, darkEnough, cv::CMP_LE);
    cv::Mat inRange;
    cv::inRange(gray, params.grayMin, params.grayMax, inRange);
    cv::bitwise_and(mask, darkEnough, mask);
    cv::bitwise_and(mask, inRange, mask);
    return componentsOf(mask, gray, params.minArea, params.maxArea, params.minFill, params.maxAspectRatio,
                        DefectPolarity::Dark);
}

cv::Mat DatabaseComparator::draw(const cv::Mat& patch, const std::vector<DefectCandidate>& defects) const {
    cv::Mat view = toBgr(patch);
    for (const auto& defect : defects) {
        cv::Scalar color(0, 0, 255);
        if (defect.polarity == DefectPolarity::Dark) {
            color = cv::Scalar(255, 80, 0);
        } else if (defect.polarity == DefectPolarity::Mixed) {
            color = cv::Scalar(0, 200, 255);
        } else if (defect.polarity == DefectPolarity::Unknown) {
            color = cv::Scalar(180, 180, 180);
        }
        cv::rectangle(view, defect.bbox, color, 1);
    }
    return view;
}

}  // namespace cutdie
