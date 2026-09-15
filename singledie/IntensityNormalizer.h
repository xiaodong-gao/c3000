#pragma once
#include <opencv2/opencv.hpp>
#include "BackgroundEstimator.h"

namespace singledie {

enum class IntensityMethod {
    Divide = 0,
    Subtract = 1,
    SubtractDivide = 2
};

struct IntensityNormalizeParams {
    BackgroundParams background;
    IntensityMethod method = IntensityMethod::Divide;

    float minBackground = 1e-6f;

    bool robustPercentile = true;
    float lowPercentile = 1.0f;
    float highPercentile = 99.0f;

    // Used only for Subtract mode. Output is centered around this value in [0,1].
    float subtractCenter = 0.5f;
};

class IntensityNormalizer {
public:
    explicit IntensityNormalizer(const IntensityNormalizeParams& params = {});
    // normalReference is mandatory because its value depends on the selected
    // method and, for robust scaling, on the current image.
    bool process(const cv::Mat& input, cv::Mat& background32f,
                 cv::Mat& normalized32f,
                 float& normalReference,
                 const cv::Mat& validMask = cv::Mat()) const;
private:
    bool toGray32F(const cv::Mat& input, cv::Mat& gray32f) const;
    void percentileRange(const cv::Mat& image32f, const cv::Mat& validMask,
                         float& low, float& high) const;
    IntensityNormalizeParams params_;
};

}  // namespace singledie
