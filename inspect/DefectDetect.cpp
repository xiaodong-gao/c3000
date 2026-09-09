#include "DefectDetect.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
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

}  // namespace

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

    const int openSize = oddSize(params.openSize, 1);
    const int closeSize = oddSize(params.closeSize, 1);
    if (openSize >= 3) {
        const cv::Mat openK =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(openSize, openSize));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, openK);
    }
    if (closeSize >= 3) {
        const cv::Mat closeK =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closeSize, closeSize));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, closeK);
    }

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

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    cv::Mat filtered = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int i = 1; i < count; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int shortSide = std::min(width, height);
        const double aspect = shortSide > 0
                                  ? static_cast<double>(std::max(width, height)) / shortSide
                                  : 0.0;
        if (area < params.minArea) {
            continue;
        }
        if (params.maxArea > 0 && area > params.maxArea) {
            continue;
        }
        if (width < params.minWidth || height < params.minHeight) {
            continue;
        }
        const bool largeBlob = area >= 500;
        if (!largeBlob && aspect > params.maxAspectRatio) {
            continue;
        }
        filtered.setTo(255, labels == i);
    }
    return filtered;
}

std::vector<DefectCandidate> extractDefectCandidates(const cv::Mat& blobMask,
                                                     const cv::Mat& l2Difference) {
    std::vector<DefectCandidate> candidates;
    if (blobMask.empty()) {
        return candidates;
    }

    std::string error;
    const cv::Mat l2 = l2Difference.empty() ? cv::Mat() : asFloatL2(l2Difference, error);

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
        c.area = stats.at<int>(i, cv::CC_STAT_AREA);
        c.bbox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                          stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        c.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));

        if (!l2.empty() && c.bbox.area() > 0) {
            double sum = 0.0;
            double peak = 0.0;
            int n = 0;
            const cv::Rect roi = c.bbox & cv::Rect(0, 0, l2.cols, l2.rows);
            for (int y = roi.y; y < roi.y + roi.height; ++y) {
                const int* labelRow = labels.ptr<int>(y);
                const float* l2Row = l2.ptr<float>(y);
                for (int x = roi.x; x < roi.x + roi.width; ++x) {
                    if (labelRow[x] != i) {
                        continue;
                    }
                    const double v = static_cast<double>(l2Row[x]);
                    sum += v;
                    peak = std::max(peak, v);
                    ++n;
                }
            }
            if (n > 0) {
                c.meanL2 = sum / static_cast<double>(n);
                c.maxL2 = peak;
            }
        }
        candidates.push_back(c);
    }
    return candidates;
}

cv::Mat drawDefectOverlay(const cv::Mat& base, const std::vector<DefectCandidate>& candidates) {
    cv::Mat overlay = toBgr(base);
    if (overlay.empty()) {
        return overlay;
    }
    for (const DefectCandidate& c : candidates) {
        cv::rectangle(overlay, c.bbox, cv::Scalar(0, 0, 255), 2);
        const cv::Point origin(c.bbox.x, std::max(0, c.bbox.y - 4));
        cv::putText(overlay, std::to_string(c.id), origin, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    return overlay;
}

DefectDetectOutput detectDefects(const cv::Mat& l2Difference, const cv::Mat& overlayBase,
                                 const DefectDetectParams& params) {
    DefectDetectOutput out;
    std::string error;
    if (asFloatL2(l2Difference, error).empty()) {
        out.error = error;
        return out;
    }

    out.adaptiveMask = adaptiveThresholdDiff(l2Difference, params.adaptive);
    out.morphMask = morphologicalClean(out.adaptiveMask, params.morph);
    out.blobMask = blobFilter(out.morphMask, params.blob);
    out.candidates = extractDefectCandidates(out.blobMask, l2Difference);
    out.overlay = drawDefectOverlay(overlayBase.empty() ? l2Difference : overlayBase, out.candidates);
    out.ok = true;
    return out;
}
