#include "DefectDetect.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

int oddSize(int size, int minimum) {
    size = std::max(minimum, size);
    if ((size % 2) == 0) {
        ++size;
    }
    return size;
}

cv::Mat toBgr(const cv::Mat& src) {
    if (src.empty()) {
        return {};
    }
    if (src.channels() == 3) {
        return src;
    }
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

cv::Mat asFloatL2(const cv::Mat& src, std::string& error) {
    if (src.empty()) {
        error = "L2 difference is empty";
        return {};
    }
    cv::Mat l2;
    if (src.type() == CV_32FC1) {
        l2 = src.isContinuous() ? src : src.clone();
    } else if (src.channels() == 1) {
        src.convertTo(l2, CV_32F);
    } else {
        error = "adaptive threshold expects a single-channel L2 map";
        return {};
    }
    return l2;
}

cv::Mat toGrayFloat(const cv::Mat& src) {
    cv::Mat gray;
    if (src.channels() == 1) {
        gray = src;
    } else if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return {};
    }
    cv::Mat f32;
    gray.convertTo(f32, CV_32F);
    return f32;
}

double robustMedianMadThreshold(const cv::Mat& l2, double k, double minAbs) {
    const int n = l2.rows * l2.cols;
    if (n <= 0) {
        return minAbs;
    }
    std::vector<float> vals(static_cast<size_t>(n));
    std::memcpy(vals.data(), l2.ptr<float>(0), static_cast<size_t>(n) * sizeof(float));

    const size_t mid = static_cast<size_t>(n - 1) / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    const double median = static_cast<double>(vals[mid]);

    for (float& v : vals) {
        v = std::abs(v - static_cast<float>(median));
    }
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    const double mad = static_cast<double>(vals[mid]);
    if (mad < 1e-9) {
        return minAbs;
    }
    return std::max(minAbs, median + k * 1.4826 * mad);
}

double resolvePolarityT(const cv::Mat& signedDifference, const DefectDetectParams& params) {
    if (params.signedDiff.T > 0.0) {
        return params.signedDiff.T;
    }
    cv::Mat absSigned;
    cv::absdiff(signedDifference, cv::Scalar(0), absSigned);
    return robustMedianMadThreshold(absSigned, params.adaptive.k, params.adaptive.minAbs);
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

cv::Scalar polarityColor(DefectPolarity polarity) {
    switch (polarity) {
        case DefectPolarity::Bright:
            return cv::Scalar(0, 0, 255);
        case DefectPolarity::Dark:
            return cv::Scalar(255, 80, 0);
        case DefectPolarity::Mixed:
            return cv::Scalar(0, 200, 255);
        default:
            return cv::Scalar(180, 180, 180);
    }
}

}  // namespace

const char* defectPolarityName(DefectPolarity polarity) {
    switch (polarity) {
        case DefectPolarity::Bright:
            return "bright";
        case DefectPolarity::Dark:
            return "dark";
        case DefectPolarity::Mixed:
            return "mixed";
        default:
            return "unknown";
    }
}

double signedPolarityThreshold(const cv::Mat& signedDifference, const DefectDetectParams& params) {
    if (signedDifference.empty()) {
        return params.signedDiff.T > 0.0 ? params.signedDiff.T : params.adaptive.minAbs;
    }
    return resolvePolarityT(signedDifference, params);
}

double signedDarkPolarityThreshold(const DefectDetectParams& params, double brightT) {
    return params.signedDiff.darkT > 0.0 ? params.signedDiff.darkT : brightT;
}

void clearMaskBorder(cv::Mat& mask, int margin) {
    if (mask.empty() || margin <= 0) {
        return;
    }
    const int m = std::min({margin, std::max(0, mask.rows / 4), std::max(0, mask.cols / 4)});
    if (m <= 0) {
        return;
    }
    mask.rowRange(0, m).setTo(0);
    mask.rowRange(mask.rows - m, mask.rows).setTo(0);
    mask.colRange(0, m).setTo(0);
    mask.colRange(mask.cols - m, mask.cols).setTo(0);
}

cv::Mat darkSpotAssistMask(const cv::Mat& dataNormalized, const cv::Mat& signedDifference,
                           const DarkSpotParams& params, double signedGate) {
    if (!params.enableBlackHat || dataNormalized.empty()) {
        return {};
    }
    const cv::Mat dataF = toGrayFloat(dataNormalized);
    if (dataF.empty()) {
        return {};
    }
    cv::Mat gray8;
    dataF.convertTo(gray8, CV_8U);
    const int ksize = oddSize(params.kernel, 3);
    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::Mat hat;
    cv::morphologyEx(gray8, hat, cv::MORPH_BLACKHAT, kernel);
    cv::Mat hatMask;
    cv::compare(hat, params.minResponse, hatMask, cv::CMP_GT);
    if (!signedDifference.empty() && signedDifference.size() == hatMask.size() &&
        signedDifference.type() == CV_32FC1) {
        const double gate = signedGate > 0.0 ? std::max(1.0, signedGate * 0.5) : 1.0;
        cv::Mat signedDark;
        cv::compare(signedDifference, -gate, signedDark, cv::CMP_LT);
        cv::bitwise_and(hatMask, signedDark, hatMask);
    }
    return hatMask;
}

cv::Mat signedDifferenceMap(const cv::Mat& templateNormalized, const cv::Mat& dataNormalized) {
    if (templateNormalized.empty() || dataNormalized.empty() ||
        templateNormalized.size() != dataNormalized.size()) {
        return {};
    }
    const cv::Mat templateF = toGrayFloat(templateNormalized);
    const cv::Mat dataF = toGrayFloat(dataNormalized);
    if (templateF.empty() || dataF.empty()) {
        return {};
    }
    return dataF - templateF;
}

cv::Mat polarityMasks(const cv::Mat& signedDifference, double brightT, double darkT,
                      cv::Mat& brightMask, cv::Mat& darkMask) {
    if (signedDifference.empty() || signedDifference.type() != CV_32FC1) {
        brightMask.release();
        darkMask.release();
        return {};
    }
    cv::compare(signedDifference, brightT, brightMask, cv::CMP_GT);
    cv::compare(signedDifference, -darkT, darkMask, cv::CMP_LT);
    cv::Mat polarity;
    cv::bitwise_or(brightMask, darkMask, polarity);
    return polarity;
}

cv::Mat polarityMasks(const cv::Mat& signedDifference, double T, cv::Mat& brightMask,
                      cv::Mat& darkMask) {
    return polarityMasks(signedDifference, T, T, brightMask, darkMask);
}

cv::Mat fuseDifference(const cv::Mat& l2Difference, const cv::Mat& polarityMask) {
    if (l2Difference.empty()) {
        return {};
    }
    cv::Mat fused = l2Difference.clone();
    if (!polarityMask.empty() && polarityMask.size() == fused.size()) {
        fused.setTo(0, polarityMask == 0);
    }
    return fused;
}

void applyDefectPolarity(const cv::Mat& l2Difference, const cv::Mat& templateNormalized,
                         const cv::Mat& dataNormalized, const DefectDetectParams& params,
                         cv::Mat& signedDifference, cv::Mat& brightMask, cv::Mat& darkMask,
                         cv::Mat& polarityMask, cv::Mat& fusedDifference, double& polarityT,
                         double& polarityDarkT) {
    signedDifference = signedDifferenceMap(templateNormalized, dataNormalized);
    polarityT = 0.0;
    polarityDarkT = 0.0;
    if (signedDifference.empty()) {
        fusedDifference = l2Difference.clone();
        brightMask.release();
        darkMask.release();
        polarityMask.release();
        return;
    }
    polarityT = resolvePolarityT(signedDifference, params);
    polarityDarkT = signedDarkPolarityThreshold(params, polarityT);
    polarityMask = polarityMasks(signedDifference, polarityT, polarityDarkT, brightMask, darkMask);
    const cv::Mat darkAssist =
        darkSpotAssistMask(dataNormalized, signedDifference, params.darkSpot, polarityDarkT);
    if (!darkAssist.empty() && darkAssist.size() == darkMask.size()) {
        cv::bitwise_or(darkMask, darkAssist, darkMask);
        cv::bitwise_or(polarityMask, darkMask, polarityMask);
    }
    clearMaskBorder(brightMask, params.blob.borderMargin);
    clearMaskBorder(darkMask, params.blob.borderMargin);
    clearMaskBorder(polarityMask, params.blob.borderMargin);
    fusedDifference = fuseDifference(l2Difference, polarityMask);
}

cv::Mat mergeMorphWithDarkChannel(const cv::Mat& adaptiveMask, const cv::Mat& darkMask,
                                  const MorphologicalParams& morph) {
    cv::Mat morphMask = morphologicalClean(adaptiveMask, morph);
    if (darkMask.empty() || darkMask.size() != morphMask.size() || morph.darkOpenSize < 3) {
        return morphMask;
    }
    MorphologicalParams darkMorph = morph;
    darkMorph.openSize = morph.darkOpenSize;
    const cv::Mat darkClean = morphologicalClean(darkMask, darkMorph);
    if (!darkClean.empty()) {
        cv::bitwise_or(morphMask, darkClean, morphMask);
    }
    return morphMask;
}

cv::Mat adaptiveThresholdDiff(const cv::Mat& l2Difference, const AdaptiveThresholdParams& params) {
    std::string error;
    const cv::Mat l2 = asFloatL2(l2Difference, error);
    if (l2.empty()) {
        return {};
    }

    int block = params.blockSize;
    if (block <= 0) {
        block = std::max(15, std::min(l2.cols, l2.rows) / 6);
    }
    block = oddSize(block, 3);
    cv::Mat mean;
    cv::Mat meanSq;
    cv::Mat sq;
    cv::GaussianBlur(l2, mean, cv::Size(block, block), 0.0);
    cv::multiply(l2, l2, sq);
    cv::GaussianBlur(sq, meanSq, cv::Size(block, block), 0.0);

    cv::Mat var = meanSq - mean.mul(mean);
    cv::max(var, 0.0, var);
    cv::Mat localStd;
    cv::sqrt(var, localStd);

    cv::Mat thresh = mean + params.k * localStd;
    cv::max(thresh, params.minAbs, thresh);

    cv::Mat localMask;
    cv::compare(l2, thresh, localMask, cv::CMP_GT);

    const double globalT = robustMedianMadThreshold(l2, params.k, params.minAbs);
    cv::Mat globalMask;
    cv::compare(l2, globalT, globalMask, cv::CMP_GT);

    cv::Mat mask;
    cv::bitwise_or(localMask, globalMask, mask);
    return mask;
}

cv::Mat morphologicalClean(const cv::Mat& binary, const MorphologicalParams& params) {
    if (binary.empty()) {
        return {};
    }
    cv::Mat mask;
    if (binary.type() != CV_8UC1) {
        binary.convertTo(mask, CV_8U);
    } else {
        mask = binary.clone();
    }

    const auto apply = [&](int size, cv::MorphTypes op) {
        const int ksize = oddSize(size, 1);
        if (ksize < 3) {
            return;
        }
        const cv::Mat kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::morphologyEx(mask, mask, op, kernel);
    };

    apply(params.openSize, cv::MORPH_OPEN);
    apply(params.closeSize, cv::MORPH_CLOSE);
    apply(params.dilateSize, cv::MORPH_DILATE);
    apply(params.erodeSize, cv::MORPH_ERODE);
    apply(params.removeNoiseSize, cv::MORPH_OPEN);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat filled = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(filled, contours, -1, cv::Scalar(255), cv::FILLED);
    return filled;
}

cv::Mat blobFilter(const cv::Mat& binary, const BlobFilterParams& params) {
    if (binary.empty()) {
        return {};
    }
    cv::Mat mask = binary;
    if (binary.type() != CV_8UC1) {
        binary.convertTo(mask, CV_8U);
    }

    const std::vector<DefectCandidate> all = extractDefectCandidates(mask, {});
    const std::vector<DefectCandidate> kept = filterDefects(all, params, {}, mask.size());
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    return maskFromCandidates(mask.size(), kept, labels);
}

std::vector<DefectCandidate> extractDefectCandidates(const cv::Mat& blobMask,
                                                     const cv::Mat& l2Difference,
                                                     const cv::Mat& signedDifference,
                                                     const cv::Mat& brightMask,
                                                     const cv::Mat& darkMask) {
    std::vector<DefectCandidate> candidates;
    if (blobMask.empty()) {
        return candidates;
    }

    std::string error;
    const cv::Mat l2 = l2Difference.empty() ? cv::Mat() : asFloatL2(l2Difference, error);
    const cv::Mat signedMap =
        (!signedDifference.empty() && signedDifference.type() == CV_32FC1) ? signedDifference
                                                                           : cv::Mat();

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count =
        cv::connectedComponentsWithStats(blobMask, labels, stats, centroids, 8, CV_32S);
    candidates.reserve(static_cast<size_t>(std::max(0, count - 1)));

    int id = 0;
    for (int i = 1; i < count; ++i) {
        DefectCandidate c;
        c.id = ++id;
        c.labelId = i;
        c.area = stats.at<int>(i, cv::CC_STAT_AREA);
        c.bbox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                          stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        c.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));

        const cv::Rect roi = c.bbox & cv::Rect(0, 0, blobMask.cols, blobMask.rows);
        if (roi.area() <= 0) {
            candidates.push_back(c);
            continue;
        }

        double sumL2 = 0.0;
        double peakL2 = 0.0;
        double sumSigned = 0.0;
        double peakSigned = -1e9;
        double minSigned = 1e9;
        int n = 0;
        for (int y = roi.y; y < roi.y + roi.height; ++y) {
            const int* labelRow = labels.ptr<int>(y);
            const float* l2Row = l2.empty() ? nullptr : l2.ptr<float>(y);
            const float* signedRow = signedMap.empty() ? nullptr : signedMap.ptr<float>(y);
            const unsigned char* brightRow =
                brightMask.empty() ? nullptr : brightMask.ptr<unsigned char>(y);
            const unsigned char* darkRow =
                darkMask.empty() ? nullptr : darkMask.ptr<unsigned char>(y);
            for (int x = roi.x; x < roi.x + roi.width; ++x) {
                if (labelRow[x] != i) {
                    continue;
                }
                ++n;
                if (l2Row) {
                    const double v = static_cast<double>(l2Row[x]);
                    sumL2 += v;
                    peakL2 = std::max(peakL2, v);
                }
                if (signedRow) {
                    const double v = static_cast<double>(signedRow[x]);
                    sumSigned += v;
                    peakSigned = std::max(peakSigned, v);
                    minSigned = std::min(minSigned, v);
                }
                if (brightRow && brightRow[x]) {
                    ++c.brightArea;
                }
                if (darkRow && darkRow[x]) {
                    ++c.darkArea;
                }
            }
        }
        if (n > 0) {
            c.meanL2 = sumL2 / static_cast<double>(n);
            c.maxL2 = peakL2;
            if (!signedMap.empty()) {
                c.meanSigned = sumSigned / static_cast<double>(n);
                c.maxSigned = peakSigned;
                c.minSigned = minSigned;
            }
        }
        c.polarity = classifyPolarity(c.brightArea, c.darkArea);
        candidates.push_back(c);
    }
    return candidates;
}

std::vector<DefectCandidate> filterDefects(const std::vector<DefectCandidate>& candidates,
                                           const BlobFilterParams& blob,
                                           const DefectFilterParams& filter,
                                           cv::Size imageSize) {
    std::vector<DefectCandidate> kept;
    kept.reserve(candidates.size());
    int id = 0;
    for (DefectCandidate c : candidates) {
        int minArea = blob.minArea;
        if (c.polarity == DefectPolarity::Bright && blob.minBrightArea > 0) {
            minArea = std::max(minArea, blob.minBrightArea);
        }
        if (c.area < minArea) {
            continue;
        }
        if (blob.maxArea > 0 && c.area > blob.maxArea) {
            continue;
        }
        if (c.bbox.width < blob.minWidth || c.bbox.height < blob.minHeight) {
            continue;
        }
        const int shortSide = std::min(c.bbox.width, c.bbox.height);
        const double aspect = shortSide > 0
                                  ? static_cast<double>(std::max(c.bbox.width, c.bbox.height)) /
                                        static_cast<double>(shortSide)
                                  : 0.0;
        if (aspect > blob.maxAspectRatio) {
            continue;
        }
        if (imageSize.width > 0 && imageSize.height > 0 && blob.borderMargin > 0) {
            const int m = blob.borderMargin;
            if (c.bbox.x <= m || c.bbox.y <= m ||
                c.bbox.x + c.bbox.width >= imageSize.width - m ||
                c.bbox.y + c.bbox.height >= imageSize.height - m) {
                continue;
            }
        }
        if (!keepPolarity(c.polarity, filter)) {
            continue;
        }
        if (c.meanL2 < filter.minMeanL2) {
            continue;
        }
        if (std::abs(c.meanSigned) < filter.minAbsMeanSigned) {
            continue;
        }
        c.id = ++id;
        kept.push_back(c);
    }
    return kept;
}

cv::Mat maskFromCandidates(const cv::Size& size, const std::vector<DefectCandidate>& candidates,
                           const cv::Mat& labels) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    if (labels.empty() || labels.size() != size || candidates.empty()) {
        return mask;
    }
    std::unordered_set<int> keep;
    keep.reserve(candidates.size());
    for (const auto& c : candidates) {
        keep.insert(c.labelId);
    }
    for (int y = 0; y < labels.rows; ++y) {
        const int* row = labels.ptr<int>(y);
        unsigned char* out = mask.ptr<unsigned char>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (keep.count(row[x]) != 0) {
                out[x] = 255;
            }
        }
    }
    return mask;
}

cv::Mat drawDefectOverlay(const cv::Mat& base, const std::vector<DefectCandidate>& candidates) {
    cv::Mat overlay = toBgr(base);
    if (overlay.empty()) {
        return overlay;
    }
    const cv::Rect bounds(0, 0, overlay.cols, overlay.rows);
    for (const DefectCandidate& c : candidates) {
        const cv::Scalar color = polarityColor(c.polarity);
        const int minSide = 16;
        cv::Rect vis = c.bbox;
        if (vis.width < minSide || vis.height < minSide) {
            const int cx = vis.x + vis.width / 2;
            const int cy = vis.y + vis.height / 2;
            vis.x = cx - minSide / 2;
            vis.y = cy - minSide / 2;
            vis.width = minSide;
            vis.height = minSide;
        }
        vis &= bounds;
        if (vis.area() > 0) {
            cv::rectangle(overlay, vis, color, 2);
        } else {
            cv::rectangle(overlay, c.bbox, color, 2);
        }
        const cv::Point center(cvRound(c.centroid.x), cvRound(c.centroid.y));
        if (bounds.contains(center)) {
            cv::circle(overlay, center, 4, color, cv::FILLED, cv::LINE_AA);
        }
        const cv::Point origin(c.bbox.x, std::max(0, c.bbox.y - 4));
        const std::string label = std::to_string(c.id) + " " + defectPolarityName(c.polarity);
        cv::putText(overlay, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
    }
    return overlay;
}

DefectDetectOutput detectDefects(const cv::Mat& l2Difference, const cv::Mat& overlayBase,
                                 const DefectDetectParams& params,
                                 const cv::Mat& templateNormalized,
                                 const cv::Mat& dataNormalized) {
    DefectDetectOutput out;
    std::string error;
    const cv::Mat l2 = asFloatL2(l2Difference, error);
    if (l2.empty()) {
        out.error = error;
        return out;
    }

    out.signedDifference = {};
    applyDefectPolarity(l2, templateNormalized, dataNormalized, params, out.signedDifference,
                        out.brightMask, out.darkMask, out.polarityMask, out.fusedDifference,
                        out.polarityT, out.polarityDarkT);

    out.adaptiveMask = adaptiveThresholdDiff(out.fusedDifference, params.adaptive);
    out.morphMask = mergeMorphWithDarkChannel(out.adaptiveMask, out.darkMask, params.morph);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(out.morphMask, labels, stats, centroids, 8, CV_32S);
    const std::vector<DefectCandidate> raw = extractDefectCandidates(
        out.morphMask, l2, out.signedDifference, out.brightMask, out.darkMask);
    out.candidates = filterDefects(raw, params.blob, params.filter, out.morphMask.size());
    out.blobMask = maskFromCandidates(out.morphMask.size(), out.candidates, labels);
    out.overlay =
        drawDefectOverlay(overlayBase.empty() ? l2Difference : overlayBase, out.candidates);
    out.ok = true;
    return out;
}
