#include "BrightDarkDetector.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace singledie {

namespace {

cv::Mat toGray8U(const cv::Mat& source) {
    if (source.empty()) return cv::Mat();

    cv::Mat gray;
    if (source.channels() == 1)
        gray = source;
    else if (source.channels() == 3)
        cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    else if (source.channels() == 4)
        cv::cvtColor(source, gray, cv::COLOR_BGRA2GRAY);
    else
        return cv::Mat();

    if (gray.depth() == CV_8U) return gray;

    cv::Mat gray8u;
    if (gray.depth() == CV_16U) {
        gray.convertTo(gray8u, CV_8U, 1.0 / 257.0);
        return gray8u;
    }
    if (gray.depth() == CV_32F || gray.depth() == CV_64F) {
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::minMaxLoc(gray, &minValue, &maxValue);
        if (maxValue <= 1.5)
            gray.convertTo(gray8u, CV_8U, 255.0);
        else
            cv::normalize(gray, gray8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        return gray8u;
    }
    return cv::Mat();
}

cv::Mat toGray32F(const cv::Mat& source) {
    if (source.empty()) return cv::Mat();

    cv::Mat gray;
    if (source.channels() == 1)
        gray = source;
    else if (source.channels() == 3)
        cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    else if (source.channels() == 4)
        cv::cvtColor(source, gray, cv::COLOR_BGRA2GRAY);
    else
        return cv::Mat();

    cv::Mat gray32f;
    switch (gray.depth()) {
    case CV_8U:
        gray.convertTo(gray32f, CV_32F, 1.0 / 255.0);
        return gray32f;
    case CV_16U:
        gray.convertTo(gray32f, CV_32F, 1.0 / 65535.0);
        return gray32f;
    case CV_32F:
        return gray;
    case CV_64F:
        gray.convertTo(gray32f, CV_32F);
        return gray32f;
    default:
        return cv::Mat();
    }
}

cv::SimpleBlobDetector::Params makeBlobParams(bool faint) {
    cv::SimpleBlobDetector::Params params;
    params.minThreshold = 2.0f;
    params.maxThreshold = 220.0f;
    params.thresholdStep = 3.0f;
    params.minRepeatability = 2;
    params.minDistBetweenBlobs = 10.0f;
    params.filterByColor = true;
    params.blobColor = 0;
    params.filterByArea = true;
    params.minArea = faint ? 30.0f : 35.0f;
    params.maxArea = 280.0f;
    params.filterByCircularity = true;
    params.minCircularity = faint ? 0.34f : 0.38f;
    params.filterByInertia = true;
    params.minInertiaRatio = faint ? 0.24f : 0.28f;
    params.filterByConvexity = true;
    params.minConvexity = faint ? 0.62f : 0.68f;
    return params;
}

std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat& gray8u, bool faint) {
    const cv::Ptr<cv::SimpleBlobDetector> detector =
        cv::SimpleBlobDetector::create(makeBlobParams(faint));
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray8u, keypoints);
    return keypoints;
}

float localMaxResponse(const cv::Mat& response32f, int x, int y, int radius) {
    const cv::Rect bounds(0, 0, response32f.cols, response32f.rows);
    cv::Rect roi(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1);
    roi &= bounds;
    if (roi.empty()) return 0.0f;
    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(response32f(roi), &minValue, &maxValue);
    return static_cast<float>(maxValue);
}

}  // namespace

BrightDarkDetector::BrightDarkDetector(const DetectionParams& params)
    : params_(params) {}

void BrightDarkDetector::morphology(cv::Mat& mask,
                                    const cv::Mat& kernel) const {
    if (kernel.empty()) return;

    if (params_.openIterations > 0)
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel,
                         {-1,-1}, params_.openIterations);
    if (params_.closeIterations > 0)
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel,
                         {-1,-1}, params_.closeIterations);
}

void BrightDarkDetector::extract(const cv::Mat& response32f,
                                  const cv::Mat& mask,
                                  bool bright,
                                  cv::Mat& acceptedMask,
                                  std::vector<Defect>& defects) const {
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(
        mask, labels, stats, centroids, 8, CV_32S);

    acceptedMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    std::vector<unsigned char> accepted(static_cast<size_t>(count), 0);
    int acceptedCount = 0;

    for (int i = 1; i < count; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < params_.minArea || area > params_.maxArea) continue;

        cv::Rect box(
            stats.at<int>(i, cv::CC_STAT_LEFT),
            stats.at<int>(i, cv::CC_STAT_TOP),
            stats.at<int>(i, cv::CC_STAT_WIDTH),
            stats.at<int>(i, cv::CC_STAT_HEIGHT));

        const int m = std::max(0, params_.borderMargin);
        if (box.x < m ||
            box.y < m ||
            box.x + box.width > mask.cols - m ||
            box.y + box.height > mask.rows - m) {
            continue;
        }

        if (params_.maxAspectRatio > 0.0f) {
            const float aspect =
                static_cast<float>(std::max(box.width, box.height)) /
                static_cast<float>(std::max(1, std::min(box.width, box.height)));
            if (aspect > params_.maxAspectRatio) continue;
        }
        if (params_.minFillRatio > 0.0f) {
            const float fill = static_cast<float>(area) /
                static_cast<float>(std::max(1, box.width * box.height));
            if (fill < params_.minFillRatio) continue;
        }

        accepted[static_cast<size_t>(i)] = 1;
        ++acceptedCount;
    }

    if (acceptedCount == 0) return;

    const bool hasIntensity =
        !response32f.empty() && response32f.size() == mask.size() &&
        (response32f.type() == CV_32FC1 || response32f.type() == CV_8UC1);
    const double intensityScale =
        (hasIntensity && response32f.depth() == CV_8U) ? (1.0 / 255.0) : 1.0;

    for (int i = 1; i < count; ++i) {
        if (!accepted[static_cast<size_t>(i)]) continue;

        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Rect box(
            stats.at<int>(i, cv::CC_STAT_LEFT),
            stats.at<int>(i, cv::CC_STAT_TOP),
            stats.at<int>(i, cv::CC_STAT_WIDTH),
            stats.at<int>(i, cv::CC_STAT_HEIGHT));

        cv::Mat localMask;
        cv::compare(labels(box), i, localMask, cv::CMP_EQ);
        localMask.copyTo(acceptedMask(box), localMask);

        double meanResponse = 0.0;
        double maxResponse = 0.0;
        if (hasIntensity) {
            const cv::Mat roi = response32f(box);
            const cv::Scalar mean = cv::mean(roi, localMask);
            double minValue = 0.0;
            cv::minMaxLoc(roi, &minValue, &maxResponse, nullptr, nullptr,
                          localMask);
            meanResponse = mean[0] * intensityScale;
            maxResponse *= intensityScale;
        }

        Defect d;
        d.id = static_cast<int>(defects.size());
        d.bright = bright;
        d.bbox = box;
        d.area = area;
        d.center = {
            centroids.at<double>(i,0),
            centroids.at<double>(i,1)
        };
        d.meanResponse = meanResponse;
        d.maxResponse = maxResponse;
        defects.push_back(d);
    }
}

void BrightDarkDetector::extractBlobs(const cv::Mat& sourceGray8u,
                                      const cv::Mat& darkResponse32f,
                                      const cv::Mat& validMask,
                                      cv::Mat& acceptedMask,
                                      std::vector<Defect>& defects) const {
    acceptedMask = cv::Mat::zeros(darkResponse32f.size(), CV_8UC1);
    if (sourceGray8u.empty() ||
        sourceGray8u.size() != darkResponse32f.size() ||
        sourceGray8u.type() != CV_8UC1) {
        return;
    }

    cv::Mat padOk;
    if (validMask.empty()) {
        padOk = cv::Mat(sourceGray8u.size(), CV_8UC1, cv::Scalar(255));
    } else {
        cv::Mat invalid;
        cv::compare(validMask, 0, invalid, cv::CMP_EQ);
        cv::dilate(invalid, invalid,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
        cv::Mat validAwayFromPad;
        cv::bitwise_not(invalid, validAwayFromPad);
        cv::bitwise_and(validMask, validAwayFromPad, padOk);
    }

    struct Candidate {
        cv::KeyPoint keypoint;
        float maxResponse = 0.0f;
    };

    auto collect = [&](const std::vector<cv::KeyPoint>& keypoints,
                       float minSize, float maxSize, float minDark,
                       std::vector<Candidate>& output) {
        for (const cv::KeyPoint& keypoint : keypoints) {
            const int x = static_cast<int>(std::lround(keypoint.pt.x));
            const int y = static_cast<int>(std::lround(keypoint.pt.y));
            if (x < 0 || y < 0 || x >= padOk.cols || y >= padOk.rows)
                continue;
            if (padOk.at<unsigned char>(y, x) == 0)
                continue;
            if (keypoint.size < minSize || keypoint.size > maxSize)
                continue;
            const float maxResponse = localMaxResponse(darkResponse32f, x, y, 4);
            if (maxResponse < minDark)
                continue;
            output.push_back({keypoint, maxResponse});
        }
    };

    std::vector<Candidate> candidates;
    collect(detectKeypoints(sourceGray8u, false), 10.0f, 20.0f, 0.085f,
            candidates);

    if (candidates.empty()) {
        constexpr int pad = 16;
        const cv::Scalar mean = cv::mean(sourceGray8u, padOk);
        cv::Mat padded;
        cv::copyMakeBorder(sourceGray8u, padded, pad, pad, pad, pad,
                           cv::BORDER_CONSTANT, mean);
        std::vector<cv::KeyPoint> paddedKeypoints =
            detectKeypoints(padded, false);
        for (cv::KeyPoint& keypoint : paddedKeypoints) {
            keypoint.pt.x -= static_cast<float>(pad);
            keypoint.pt.y -= static_cast<float>(pad);
        }
        collect(paddedKeypoints, 10.0f, 20.0f, 0.085f, candidates);
    }

    if (candidates.empty()) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        cv::Mat enhanced;
        clahe->apply(sourceGray8u, enhanced);
        collect(detectKeypoints(enhanced, true), 10.0f, 16.0f, 0.094f,
                candidates);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& left, const Candidate& right) {
                  return left.maxResponse > right.maxResponse;
              });

    std::vector<Candidate> kept;
    kept.reserve(candidates.size());
    for (const Candidate& candidate : candidates) {
        bool overlap = false;
        for (const Candidate& other : kept) {
            const float dx = candidate.keypoint.pt.x - other.keypoint.pt.x;
            const float dy = candidate.keypoint.pt.y - other.keypoint.pt.y;
            if (dx * dx + dy * dy < 12.0f * 12.0f) {
                overlap = true;
                break;
            }
        }
        if (!overlap) kept.push_back(candidate);
    }

    for (const Candidate& candidate : kept) {
        const int radius = std::max(
            1, static_cast<int>(std::lround(candidate.keypoint.size / 2.0f)));
        const cv::Point center(
            static_cast<int>(std::lround(candidate.keypoint.pt.x)),
            static_cast<int>(std::lround(candidate.keypoint.pt.y)));
        const int x0 = std::max(0, center.x - radius);
        const int y0 = std::max(0, center.y - radius);
        const int x1 = std::min(acceptedMask.cols, center.x + radius + 1);
        const int y1 = std::min(acceptedMask.rows, center.y + radius + 1);
        if (x1 <= x0 || y1 <= y0) continue;

        const cv::Rect box(x0, y0, x1 - x0, y1 - y0);
        cv::Mat localMask = cv::Mat::zeros(box.size(), CV_8UC1);
        cv::circle(localMask, cv::Point(center.x - x0, center.y - y0), radius,
                   cv::Scalar(255), cv::FILLED, cv::LINE_8, 0);
        if (!validMask.empty())
            cv::bitwise_and(localMask, validMask(box), localMask);

        const int area = cv::countNonZero(localMask);
        if (area < 1) continue;

        const cv::Mat roiResponse = darkResponse32f(box);
        const cv::Scalar mean = cv::mean(roiResponse, localMask);
        if (mean[0] < 0.035) continue;

        double maxResponse = 0.0;
        cv::minMaxLoc(roiResponse, nullptr, &maxResponse, nullptr, nullptr,
                      localMask);

        localMask.copyTo(acceptedMask(box), localMask);

        Defect defect;
        defect.id = static_cast<int>(defects.size());
        defect.bright = false;
        defect.bbox = box;
        defect.area = area;
        defect.center = {candidate.keypoint.pt.x, candidate.keypoint.pt.y};
        defect.meanResponse = mean[0];
        defect.maxResponse = maxResponse;
        defects.push_back(defect);
    }
}

bool BrightDarkDetector::detect(const cv::Mat& normalized32f,
                                 DetectionResult& result,
                                 const cv::Mat& validMask,
                                 const cv::Mat& sourceImage) const try {
    const cv::Mat normalizedInput = normalized32f;
    const cv::Mat validityInput = validMask;
    const cv::Mat sourceInput = sourceImage;
    result = DetectionResult();
    const bool absoluteBright =
        params_.useAbsoluteBright && params_.enableBright;
    if ((!absoluteBright &&
         (normalizedInput.empty() || normalizedInput.type() != CV_32FC1)) ||
        (absoluteBright && sourceInput.empty() &&
         (normalizedInput.empty() || normalizedInput.type() != CV_32FC1)) ||
        !std::isfinite(params_.normalReference) ||
        !std::isfinite(params_.brightThreshold) ||
        !std::isfinite(params_.darkThreshold) ||
        params_.brightThreshold < 0.0f || params_.darkThreshold < 0.0f ||
        params_.morphologyKernel < 1 || params_.openIterations < 0 ||
        params_.closeIterations < 0 || params_.minArea < 1 ||
        params_.maxArea < params_.minArea || params_.borderMargin < 0 ||
        params_.maxAspectRatio < 0.0f || params_.minFillRatio < 0.0f ||
        params_.minFillRatio > 1.0f) {
        return false;
    }

    try {
        cv::Mat absoluteGray8u;
        if (absoluteBright) {
            absoluteGray8u = toGray8U(
                sourceInput.empty() ? normalizedInput : sourceInput);
            if (absoluteGray8u.empty()) return false;
        }
        const cv::Size imageSize = absoluteBright
            ? absoluteGray8u.size()
            : normalizedInput.size();
        if (!validityInput.empty() &&
            (validityInput.type() != CV_8UC1 ||
             validityInput.size() != imageSize)) {
            return false;
        }

        cv::Mat detectionImage32f;
        if (!absoluteBright) {
            if (validityInput.empty()) {
                detectionImage32f = normalizedInput;
            } else {
                detectionImage32f = cv::Mat(
                    imageSize, CV_32FC1,
                    cv::Scalar(params_.normalReference));
                normalizedInput.copyTo(detectionImage32f, validityInput);
            }
            if (!cv::checkRange(detectionImage32f)) return false;
        }

        cv::Mat sourceGray8u;
        if (params_.enableBlob && params_.enableDark) {
            sourceGray8u = toGray8U(
                sourceInput.empty() ? detectionImage32f : sourceInput);
            if (sourceGray8u.empty() || sourceGray8u.size() != imageSize) {
                return false;
            }
        }

        cv::Mat morphologyKernel;
        if ((params_.enableBright ||
             (params_.enableDark && !params_.enableBlob)) &&
            (params_.openIterations > 0 || params_.closeIterations > 0) &&
            params_.morphologyKernel > 1) {
            int k = params_.morphologyKernel;
            if ((k & 1) == 0) ++k;
            const auto maximumUsefulKernel =
                2LL * std::max(imageSize.height, imageSize.width) + 1;
            if (static_cast<long long>(k) > maximumUsefulKernel)
                return false;
            morphologyKernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(k, k));
        }

        if (params_.enableBright ||
            (params_.keepResponseMaps && !absoluteBright)) {
            cv::Mat brightResponse32f;
            if (params_.enableBright) {
                cv::Mat candidateMask;
                if (absoluteBright) {
                    const int grayThreshold = std::max(
                        0, std::min(255, static_cast<int>(std::lround(
                               params_.brightThreshold * 255.0f))));
                    cv::compare(absoluteGray8u, grayThreshold, candidateMask,
                                cv::CMP_GT);
                } else {
                    cv::subtract(detectionImage32f, params_.normalReference,
                                 brightResponse32f);
                    cv::compare(brightResponse32f, params_.brightThreshold,
                                candidateMask, cv::CMP_GT);
                }
                if (!validityInput.empty())
                    cv::bitwise_and(candidateMask, validityInput, candidateMask);
                morphology(candidateMask, morphologyKernel);
                if (!validityInput.empty())
                    cv::bitwise_and(candidateMask, validityInput, candidateMask);
                extract(absoluteBright ? absoluteGray8u : brightResponse32f,
                        candidateMask, true, result.brightMask, result.defects);
                if (absoluteBright && params_.keepResponseMaps)
                    absoluteGray8u.convertTo(brightResponse32f, CV_32F,
                                             1.0 / 255.0);
            } else if (!absoluteBright) {
                cv::subtract(detectionImage32f, params_.normalReference,
                             brightResponse32f);
            }
            if (params_.keepResponseMaps)
                result.brightResponse32f = std::move(brightResponse32f);
        }
        if (result.brightMask.empty())
            result.brightMask = cv::Mat::zeros(imageSize, CV_8UC1);

        if (params_.enableDark ||
            (params_.keepResponseMaps && !absoluteBright)) {
            cv::Mat darkResponse32f;
            cv::subtract(params_.normalReference, detectionImage32f,
                         darkResponse32f);
            if (params_.enableDark) {
                if (params_.enableBlob) {
                    extractBlobs(sourceGray8u, darkResponse32f, validityInput,
                                 result.darkMask, result.defects);
                } else {
                    cv::Mat candidateMask;
                    cv::compare(darkResponse32f, params_.darkThreshold,
                                candidateMask, cv::CMP_GT);
                    if (!validityInput.empty())
                        cv::bitwise_and(candidateMask, validityInput,
                                        candidateMask);
                    morphology(candidateMask, morphologyKernel);
                    if (!validityInput.empty())
                        cv::bitwise_and(candidateMask, validityInput,
                                        candidateMask);
                    extract(darkResponse32f, candidateMask,
                            false, result.darkMask, result.defects);
                }
            }
            if (params_.keepResponseMaps)
                result.darkResponse32f = std::move(darkResponse32f);
        }
        if (result.darkMask.empty())
            result.darkMask = cv::Mat::zeros(imageSize, CV_8UC1);

        cv::bitwise_or(result.brightMask, result.darkMask, result.defectMask);
        return true;
    } catch (const cv::Exception&) {
        result = DetectionResult();
        return false;
    } catch (const std::bad_alloc&) {
        result = DetectionResult();
        return false;
    }
} catch (const cv::Exception&) {
    result = DetectionResult();
    return false;
} catch (const std::bad_alloc&) {
    result = DetectionResult();
    return false;
}

}  // namespace singledie
