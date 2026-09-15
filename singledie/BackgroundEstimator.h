#pragma once
#include <opencv2/opencv.hpp>

namespace singledie {

enum class BackgroundMethod {
    Gaussian = 0,
    Median = 1,
    MorphOpen = 2
};

struct BackgroundParams {
    BackgroundMethod method = BackgroundMethod::Gaussian;
    int kernelSize = 51;
    double sigma = 0.0;
};

class BackgroundEstimator {
public:
    explicit BackgroundEstimator(const BackgroundParams& params = {});
    // validMask is supported by Gaussian estimation. Nonzero pixels contribute
    // through normalized convolution; Median/MorphOpen reject a nonempty mask.
    bool estimate(const cv::Mat& image32f, cv::Mat& background32f,
                  const cv::Mat& validMask = cv::Mat()) const;
private:
    BackgroundParams params_;
};

}  // namespace singledie
