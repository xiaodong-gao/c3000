#include "IntensityNormalizer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace singledie {

IntensityNormalizer::IntensityNormalizer(const IntensityNormalizeParams& params)
    : params_(params) {}

bool IntensityNormalizer::toGray32F(const cv::Mat& input, cv::Mat& gray32f) const {
    if (input.empty()) return false;

    double scale = 1.0;
    switch (input.depth()) {
    case CV_8U:  scale = 1.0 / 255.0; break;
    case CV_16U: scale = 1.0 / 65535.0; break;
    case CV_32F: scale = 1.0; break;
    case CV_64F: scale = 1.0; break;
    default: return false;
    }

    cv::Mat gray;
    try {
        if (input.channels() == 1) gray = input;
        else if (input.channels() == 3)
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else if (input.channels() == 4)
            cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
        else return false;
    } catch (const cv::Exception&) {
        return false;
    }
    if (gray.type() == CV_32FC1 && scale == 1.0) {
        gray32f = gray;
    } else {
        gray.convertTo(gray32f, CV_32F, scale);
    }
    return true;
}

void IntensityNormalizer::percentileRange(const cv::Mat& image32f,
                                           const cv::Mat& validMask,
                                           float& low, float& high) const {
    std::vector<float> values;
    values.reserve(image32f.total());

    for (int y = 0; y < image32f.rows; ++y) {
        const float* p = image32f.ptr<float>(y);
        const unsigned char* maskRow = validMask.empty()
            ? nullptr : validMask.ptr<unsigned char>(y);
        for (int x = 0; x < image32f.cols; ++x)
            if ((!maskRow || maskRow[x] != 0) && std::isfinite(p[x]))
                values.push_back(p[x]);
    }

    if (values.empty()) {
        low = 0.0f; high = 1.0f; return;
    }

    const size_t n = values.size();

    auto idx = [n](float percentile) {
        percentile = std::clamp(percentile, 0.0f, 100.0f);
        return static_cast<size_t>((percentile / 100.0f) * (n - 1));
    };

    const size_t lowIndex = idx(params_.lowPercentile);
    const size_t highIndex = idx(params_.highPercentile);

    std::nth_element(values.begin(), values.begin() + lowIndex, values.end());
    low = values[lowIndex];
    std::nth_element(values.begin(), values.begin() + highIndex, values.end());
    high = values[highIndex];
    if (high <= low) high = low + 1e-6f;
}

bool IntensityNormalizer::process(const cv::Mat& input,
                                   cv::Mat& background32f,
                                   cv::Mat& normalized32f,
                                   float& normalReference,
                                   const cv::Mat& validMask) const try {
    normalReference = std::numeric_limits<float>::quiet_NaN();

    if (&background32f == &normalized32f ||
        (!validMask.empty() &&
         (validMask.type() != CV_8UC1 || validMask.size() != input.size())) ||
        (params_.method != IntensityMethod::Divide &&
         params_.method != IntensityMethod::Subtract &&
         params_.method != IntensityMethod::SubtractDivide) ||
        !std::isfinite(params_.minBackground) || params_.minBackground <= 0.0f ||
        !std::isfinite(params_.subtractCenter) ||
        !std::isfinite(params_.lowPercentile) ||
        !std::isfinite(params_.highPercentile) ||
        params_.lowPercentile < 0.0f ||
        params_.highPercentile > 100.0f ||
        params_.lowPercentile >= params_.highPercentile) {
        return false;
    }

    cv::Mat image32f;
    if (!toGray32F(input, image32f)) return false;
    cv::Mat analysisImage32f;
    if (validMask.empty()) {
        analysisImage32f = image32f;
    } else {
        analysisImage32f = cv::Mat::zeros(image32f.size(), CV_32FC1);
        image32f.copyTo(analysisImage32f, validMask);
    }
    // Invalid-mask pixels are zeroed above, so non-finite values there do not
    // poison the valid inspection region.
    if (!cv::checkRange(analysisImage32f)) return false;

    BackgroundEstimator estimator(params_.background);
    cv::Mat computedBackground;
    if (!estimator.estimate(analysisImage32f, computedBackground, validMask))
        return false;

    // Subtract mode does not divide by the background, so clamping it would
    // bias the residual (especially for dark/zero-valued images).
    if (params_.method != IntensityMethod::Subtract)
        cv::max(computedBackground, params_.minBackground, computedBackground);

    float reference = 1.0f;
    cv::Mat computedNormalized;
    switch (params_.method) {
    case IntensityMethod::Divide:
        cv::divide(analysisImage32f, computedBackground, computedNormalized);
        reference = 1.0f;
        break;

    case IntensityMethod::Subtract:
        cv::subtract(analysisImage32f, computedBackground, computedNormalized);
        computedNormalized += params_.subtractCenter;
        reference = params_.subtractCenter;
        break;

    case IntensityMethod::SubtractDivide: {
        cv::subtract(analysisImage32f, computedBackground, computedNormalized);
        cv::divide(computedNormalized, computedBackground, computedNormalized);
        reference = 0.0f;
        break;
    }
    default:
        return false;
    }

    // Do not percentile-normalize Divide mode: keeping 1.0 as the physical
    // "normal" reference is important for symmetric bright/dark thresholds.
    if (params_.robustPercentile && params_.method != IntensityMethod::Divide) {
        float low = 0.0f, high = 1.0f;
        percentileRange(computedNormalized, validMask, low, high);
        cv::max(computedNormalized, low, computedNormalized);
        cv::min(computedNormalized, high, computedNormalized);
        computedNormalized = (computedNormalized - low) / (high - low);
        // Floating-point cancellation can produce a small value outside the
        // promised [0, 1] interval when the percentile range is nearly flat.
        cv::max(computedNormalized, 0.0f, computedNormalized);
        cv::min(computedNormalized, 1.0f, computedNormalized);
        reference = std::clamp((reference - low) / (high - low), 0.0f, 1.0f);
    }

    if (!cv::checkRange(computedNormalized)) return false;
    background32f = std::move(computedBackground);
    normalized32f = std::move(computedNormalized);
    normalReference = reference;
    return true;
} catch (const cv::Exception&) {
    normalReference = std::numeric_limits<float>::quiet_NaN();
    return false;
} catch (const std::bad_alloc&) {
    normalReference = std::numeric_limits<float>::quiet_NaN();
    return false;
}

}  // namespace singledie
