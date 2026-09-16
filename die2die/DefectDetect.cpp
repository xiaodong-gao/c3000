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

std::vector<DefectCandidate> extractLargeWhiteRegions(const cv::Mat& image,
                                                      const LargeWhiteParams& params) {
    std::vector<DefectCandidate> out;
    if (!params.enable || image.empty()) {
        return out;
    }
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return out;
    }
    cv::Mat gray8;
    if (gray.type() == CV_8UC1) {
        gray8 = gray;
    } else {
        gray.convertTo(gray8, CV_8U);
    }

    auto applyMorph = [&](cv::Mat& mask) {
        if (params.morphSize < 3 || mask.empty()) {
            return;
        }
        int k = params.morphSize;
        if ((k % 2) == 0) {
            ++k;
        }
        const cv::Mat kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    };

    auto extractFromMask = [&](const cv::Mat& mask, int minArea, int maxArea, double minFill,
                               double maxAspect, double minPeak, int streetBand, int borderMargin) {
        std::vector<DefectCandidate> found;
        if (mask.empty()) {
            return found;
        }
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int count =
            cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
        for (int i = 1; i < count; ++i) {
            DefectCandidate c;
            c.area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (c.area < minArea) {
                continue;
            }
            if (maxArea > 0 && c.area > maxArea) {
                continue;
            }
            c.bbox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                              stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
            if (borderMargin > 0) {
                if (c.bbox.x <= borderMargin || c.bbox.y <= borderMargin ||
                    c.bbox.x + c.bbox.width >= gray8.cols - borderMargin ||
                    c.bbox.y + c.bbox.height >= gray8.rows - borderMargin) {
                    continue;
                }
            }
            const int shortSide = std::min(c.bbox.width, c.bbox.height);
            const double aspect = shortSide > 0
                                      ? static_cast<double>(std::max(c.bbox.width, c.bbox.height)) /
                                            static_cast<double>(shortSide)
                                      : 0.0;
            if (aspect > maxAspect) {
                continue;
            }
            const double boxArea =
                static_cast<double>(c.bbox.width) * static_cast<double>(c.bbox.height);
            if (boxArea <= 0.0 || static_cast<double>(c.area) / boxArea < minFill) {
                continue;
            }
            c.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
            if (streetBand > 0 &&
                std::abs(c.centroid.y - gray8.rows * 0.5) < static_cast<double>(streetBand)) {
                continue;
            }
            if (params.maxBackground > 0.0) {
                const int m = std::max(1, params.backgroundMargin);
                cv::Rect ring(c.bbox.x - m, c.bbox.y - m, c.bbox.width + 2 * m,
                              c.bbox.height + 2 * m);
                ring &= cv::Rect(0, 0, gray8.cols, gray8.rows);
                double bgSum = 0.0;
                int bgN = 0;
                for (int y = ring.y; y < ring.y + ring.height; ++y) {
                    const unsigned char* row = gray8.ptr<unsigned char>(y);
                    for (int x = ring.x; x < ring.x + ring.width; ++x) {
                        if (c.bbox.contains(cv::Point(x, y))) {
                            continue;
                        }
                        bgSum += static_cast<double>(row[x]);
                        ++bgN;
                    }
                }
                if (bgN <= 0 || bgSum / static_cast<double>(bgN) > params.maxBackground) {
                    continue;
                }
            }
            c.labelId = i;
            c.polarity = DefectPolarity::Bright;
            c.brightArea = c.area;
            double sum = 0.0;
            double peak = 0.0;
            int n = 0;
            const cv::Rect roi = c.bbox & cv::Rect(0, 0, gray8.cols, gray8.rows);
            for (int y = roi.y; y < roi.y + roi.height; ++y) {
                const int* lab = labels.ptr<int>(y);
                const unsigned char* row = gray8.ptr<unsigned char>(y);
                for (int x = roi.x; x < roi.x + roi.width; ++x) {
                    if (lab[x] != i) {
                        continue;
                    }
                    const double v = static_cast<double>(row[x]);
                    sum += v;
                    peak = std::max(peak, v);
                    ++n;
                }
            }
            if (n > 0) {
                c.meanL2 = sum / static_cast<double>(n);
                c.maxL2 = peak;
                c.meanSigned = c.meanL2;
                c.maxSigned = peak;
            }
            if (minPeak > 0.0 && c.maxL2 < minPeak) {
                continue;
            }
            found.push_back(c);
        }
        return found;
    };

    auto appendUnique = [&](const std::vector<DefectCandidate>& extra) {
        for (const auto& w : extra) {
            bool overlap = false;
            for (const auto& e : out) {
                const cv::Rect hit = e.bbox & w.bbox;
                const int need = std::max(1, static_cast<int>(0.3 * std::min(static_cast<int>(w.bbox.area()),
                                                                             static_cast<int>(e.bbox.area()))));
                if (hit.area() >= need) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                out.push_back(w);
            }
        }
    };

    cv::Mat absMask;
    cv::compare(gray8, params.threshold, absMask, cv::CMP_GE);
    applyMorph(absMask);
    out = extractFromMask(absMask, params.minArea, params.maxArea, params.minFill,
                          params.maxAspectRatio, 0.0, 0, 0);

    if (params.tophatSize >= 3 && params.tophatMin > 0.0) {
        int k = params.tophatSize;
        if ((k % 2) == 0) {
            ++k;
        }
        const cv::Mat hatKernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::Mat hat;
        cv::morphologyEx(gray8, hat, cv::MORPH_TOPHAT, hatKernel);
        cv::Mat tmask;
        cv::compare(hat, params.tophatMin, tmask, cv::CMP_GE);
        const double floor = params.tophatFloor > 0.0 ? params.tophatFloor : 50.0;
        cv::Mat brightEnough;
        cv::compare(gray8, floor, brightEnough, cv::CMP_GE);
        cv::bitwise_and(tmask, brightEnough, tmask);
        if (params.tophatStreetBand > 0 && gray8.rows > params.tophatStreetBand * 2) {
            const int y0 = std::max(0, gray8.rows / 2 - params.tophatStreetBand);
            const int y1 = std::min(gray8.rows, gray8.rows / 2 + params.tophatStreetBand);
            tmask(cv::Rect(0, y0, tmask.cols, y1 - y0)).setTo(0);
        }
        applyMorph(tmask);
        const int tminA = params.tophatMinArea > 0 ? params.tophatMinArea : params.minArea;
        const int tmaxA = params.tophatMaxArea;
        const double tfill = params.tophatMinFill > 0.0 ? params.tophatMinFill : params.minFill;
        const double tasp =
            params.tophatMaxAspect > 0.0 ? params.tophatMaxAspect : params.maxAspectRatio;
        appendUnique(extractFromMask(tmask, tminA, tmaxA, tfill, tasp, params.tophatMinPeak, 0,
                                     params.tophatBorderMargin));
    }

    int id = 0;
    for (auto& c : out) {
        c.id = ++id;
    }
    return out;
}

std::vector<DefectCandidate> extractDarkRegions(const cv::Mat& image,
                                                const DarkRegionParams& params) {
    std::vector<DefectCandidate> out;
    if (!params.enable || image.empty() || params.blackhatSize < 3 || params.blackhatMin <= 0.0) {
        return out;
    }
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return out;
    }
    cv::Mat gray8;
    if (gray.type() == CV_8UC1) {
        gray8 = gray;
    } else {
        gray.convertTo(gray8, CV_8U);
    }

    cv::Mat valid;
    cv::compare(gray8, params.grayMin, valid, cv::CMP_GT);
    cv::Mat belowMax;
    cv::compare(gray8, params.grayMax, belowMax, cv::CMP_LT);
    cv::bitwise_and(valid, belowMax, valid);
    if (cv::countNonZero(valid) < 32) {
        return out;
    }

    std::vector<unsigned char> vals;
    vals.reserve(static_cast<size_t>(cv::countNonZero(valid)));
    for (int y = 0; y < gray8.rows; ++y) {
        const unsigned char* row = gray8.ptr<unsigned char>(y);
        const unsigned char* m = valid.ptr<unsigned char>(y);
        for (int x = 0; x < gray8.cols; ++x) {
            if (m[x]) {
                vals.push_back(row[x]);
            }
        }
    }
    if (vals.empty()) {
        return out;
    }
    const size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    const double median = static_cast<double>(vals[mid]);

    int k = params.blackhatSize;
    if ((k % 2) == 0) {
        ++k;
    }
    const cv::Mat hatKernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
    cv::Mat hat;
    cv::morphologyEx(gray8, hat, cv::MORPH_BLACKHAT, hatKernel);
    auto collect = [&](const cv::Mat& bin, int minArea, int maxArea, double minFill, double maxAspect,
                       int streetBand, int borderMargin, double minHat) {
        std::vector<DefectCandidate> found;
        if (bin.empty()) {
            return found;
        }
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int count = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
        for (int i = 1; i < count; ++i) {
            DefectCandidate c;
            c.area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (c.area < minArea) {
                continue;
            }
            if (maxArea > 0 && c.area > maxArea) {
                continue;
            }
            c.bbox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                              stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
            if (borderMargin > 0) {
                if (c.bbox.x <= borderMargin || c.bbox.y <= borderMargin ||
                    c.bbox.x + c.bbox.width >= gray8.cols - borderMargin ||
                    c.bbox.y + c.bbox.height >= gray8.rows - borderMargin) {
                    continue;
                }
            }
            const int shortSide = std::min(c.bbox.width, c.bbox.height);
            const double aspect = shortSide > 0
                                      ? static_cast<double>(std::max(c.bbox.width, c.bbox.height)) /
                                            static_cast<double>(shortSide)
                                      : 0.0;
            if (aspect > maxAspect) {
                continue;
            }
            const double boxArea =
                static_cast<double>(c.bbox.width) * static_cast<double>(c.bbox.height);
            if (boxArea <= 0.0 || static_cast<double>(c.area) / boxArea < minFill) {
                continue;
            }
            c.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
            if (streetBand > 0 &&
                std::abs(c.centroid.y - gray8.rows * 0.5) < static_cast<double>(streetBand)) {
                continue;
            }
            c.labelId = i;
            c.polarity = DefectPolarity::Dark;
            c.darkArea = c.area;
            double sum = 0.0;
            double peak = 0.0;
            double hatPeak = 0.0;
            int n = 0;
            const cv::Rect roi = c.bbox & cv::Rect(0, 0, gray8.cols, gray8.rows);
            for (int y = roi.y; y < roi.y + roi.height; ++y) {
                const int* lab = labels.ptr<int>(y);
                const unsigned char* row = gray8.ptr<unsigned char>(y);
                const unsigned char* hrow = hat.ptr<unsigned char>(y);
                for (int x = roi.x; x < roi.x + roi.width; ++x) {
                    if (lab[x] != i) {
                        continue;
                    }
                    const double v = static_cast<double>(row[x]);
                    sum += v;
                    peak = std::max(peak, 255.0 - v);
                    hatPeak = std::max(hatPeak, static_cast<double>(hrow[x]));
                    ++n;
                }
            }
            if (n > 0) {
                c.meanL2 = sum / static_cast<double>(n);
                c.maxL2 = peak;
                c.meanSigned = c.meanL2 - median;
                c.maxSigned = peak;
            }
            if (minHat > 0.0 && hatPeak < minHat) {
                continue;
            }
            found.push_back(c);
        }
        return found;
    };

    auto appendUnique = [&](const std::vector<DefectCandidate>& extra) {
        for (const auto& w : extra) {
            bool overlap = false;
            for (const auto& e : out) {
                const cv::Rect hit = e.bbox & w.bbox;
                const int need = std::max(1, static_cast<int>(0.3 * std::min(w.bbox.area(), e.bbox.area())));
                if (hit.area() >= need) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                out.push_back(w);
            }
        }
    };

    auto makeMask = [&](double hatMin, double below) {
        cv::Mat mask;
        cv::compare(hat, hatMin, mask, cv::CMP_GE);
        cv::bitwise_and(mask, valid, mask);
        cv::Mat darkEnough;
        cv::compare(gray8, median - below, darkEnough, cv::CMP_LE);
        cv::bitwise_and(mask, darkEnough, mask);
        return mask;
    };

    cv::Mat mask = makeMask(params.blackhatMin, params.belowMedian);
    out = collect(mask, params.minArea, params.maxArea, params.minFill, params.maxAspectRatio,
                  params.streetBand, 0, 0.0);

    if (params.faintBlackhatMin > 0.0) {
        cv::Mat faintMask = makeMask(params.faintBlackhatMin,
                                     params.faintBelowMedian > 0.0 ? params.faintBelowMedian
                                                                   : params.belowMedian);
        if (params.closeSize >= 3) {
            int ck = params.closeSize;
            if ((ck % 2) == 0) {
                ++ck;
            }
            const cv::Mat closeK = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ck, ck));
            cv::morphologyEx(faintMask, faintMask, cv::MORPH_CLOSE, closeK);
        }
        appendUnique(collect(faintMask, params.faintMinArea, params.maxArea, params.faintMinFill,
                             params.faintMaxAspect, params.streetBand, params.borderMargin, 0.0));
        if (params.faintSmallMinArea > 0 && params.faintSmallHatMin > 0.0) {
            appendUnique(collect(faintMask, params.faintSmallMinArea, params.faintSmallMaxArea,
                                 params.faintSmallMinFill, params.faintSmallMaxAspect,
                                 params.streetBand, params.borderMargin, params.faintSmallHatMin));
        }
    }

    if (params.edgeMinArea > 0 && params.edgeBrightMin > 0) {
        cv::Mat brightCut;
        cv::compare(gray8, 230, brightCut, cv::CMP_GE);
        cv::Mat black;
        cv::compare(gray8, 10, black, cv::CMP_LE);
        const int nBright = cv::countNonZero(brightCut);
        const int nDark = cv::countNonZero(black);
        bool bottomCut = false;
        bool topCut = false;
        if (nBright >= params.edgeBrightMin && nDark >= params.edgeDarkMin && gray8.rows >= 8) {
            const int yLo = std::min(gray8.rows - 1, std::max(1, static_cast<int>(0.55 * gray8.rows)));
            const int yHi = std::max(1, std::min(gray8.rows - 1, static_cast<int>(0.45 * gray8.rows)));
            const int nBlackLo = cv::countNonZero(black.rowRange(yLo, gray8.rows));
            const int nBlackHi = cv::countNonZero(black.rowRange(0, yHi));
            const double frac = params.edgeBlackFrac > 0.0 ? params.edgeBlackFrac : 0.65;
            bottomCut = nBlackLo >= static_cast<int>(frac * nDark);
            topCut = nBlackHi >= static_cast<int>(frac * nDark);
        }
        if (bottomCut || topCut) {
            cv::Mat edgeValid;
            cv::compare(gray8, params.edgeGrayMin, edgeValid, cv::CMP_GT);
            cv::Mat edgeBelowMax;
            cv::compare(gray8, params.edgeGrayMax, edgeBelowMax, cv::CMP_LT);
            cv::bitwise_and(edgeValid, edgeBelowMax, edgeValid);
            cv::Mat edgeMask;
            cv::compare(hat, params.edgeBlackhatMin, edgeMask, cv::CMP_GE);
            cv::bitwise_and(edgeMask, edgeValid, edgeMask);
            cv::Mat edgeDark;
            cv::compare(gray8, median - params.edgeBelowMedian, edgeDark, cv::CMP_LE);
            cv::bitwise_and(edgeMask, edgeDark, edgeMask);
            if (params.edgeNearCut > 0) {
                int k = params.edgeNearCut * 2 + 1;
                if ((k % 2) == 0) {
                    ++k;
                }
                const cv::Mat nearK =
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
                cv::Mat nearCut;
                cv::dilate(brightCut, nearCut, nearK);
                cv::bitwise_and(edgeMask, nearCut, edgeMask);
            }
            appendUnique(collect(edgeMask, params.edgeMinArea, params.edgeMaxArea, params.edgeMinFill,
                                 params.edgeMaxAspect, 0, 0, 0.0));
        }
    }

    int id = 0;
    for (auto& c : out) {
        c.id = ++id;
    }
    return out;
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
            const bool touches = c.bbox.x <= m || c.bbox.y <= m ||
                                 c.bbox.x + c.bbox.width >= imageSize.width - m ||
                                 c.bbox.y + c.bbox.height >= imageSize.height - m;
            // 贴边长条是切割/对位伪差；贴边的紧凑黑点（如 die 顶上的小黑点）要留下
            if (touches && aspect > blob.borderStripAspect) {
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
            cv::rectangle(overlay, vis, color, 1);
        } else {
            cv::rectangle(overlay, c.bbox, color, 1);
        }
        const cv::Point center(cvRound(c.centroid.x), cvRound(c.centroid.y));
        if (bounds.contains(center)) {
            cv::circle(overlay, center, 1, color, cv::FILLED, cv::LINE_AA);
        }
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
