#include "BackgroundEstimator.h"
#include <cmath>
#include <limits>
#include <new>
#include <utility>

namespace singledie {

BackgroundEstimator::BackgroundEstimator(const BackgroundParams& params)
    : params_(params) {}

bool BackgroundEstimator::estimate(const cv::Mat& image32f,
                                   cv::Mat& background32f,
                                   const cv::Mat& validMask) const {
    if (image32f.empty() || image32f.type() != CV_32FC1 ||
        (!validMask.empty() &&
         (validMask.type() != CV_8UC1 || validMask.size() != image32f.size())) ||
        params_.kernelSize < 1 || !std::isfinite(params_.sigma) ||
        params_.sigma < 0.0) {
        return false;
    }

    int k = std::max(3, params_.kernelSize);
    if ((k & 1) == 0) ++k;

    cv::Mat computedBackground;
    try {
        switch (params_.method) {
        case BackgroundMethod::Gaussian:
            if (validMask.empty()) {
                cv::GaussianBlur(image32f, computedBackground, cv::Size(k,k),
                                 params_.sigma, params_.sigma,
                                 cv::BORDER_REPLICATE);
            } else {
                cv::Mat binaryMask;
                cv::compare(validMask, 0, binaryMask, cv::CMP_NE);
                cv::Mat mask32f;
                binaryMask.convertTo(mask32f, CV_32F, 1.0 / 255.0);

                cv::Mat weightedImage = cv::Mat::zeros(
                    image32f.size(), CV_32FC1);
                image32f.copyTo(weightedImage, binaryMask);
                cv::Mat blurredWeights;
                cv::GaussianBlur(weightedImage, computedBackground,
                                 cv::Size(k,k), params_.sigma, params_.sigma,
                                 cv::BORDER_REPLICATE);
                cv::GaussianBlur(mask32f, blurredWeights, cv::Size(k,k),
                                 params_.sigma, params_.sigma,
                                 cv::BORDER_REPLICATE);
                cv::max(blurredWeights, std::numeric_limits<float>::epsilon(),
                        blurredWeights);
                cv::divide(computedBackground, blurredWeights,
                           computedBackground);
            }
            break;
        case BackgroundMethod::Median:
            // OpenCV supports floating-point median filtering only for 3x3 and 5x5.
            if (!validMask.empty() || k > 5) return false;
            cv::medianBlur(image32f, computedBackground, k);
            break;
        case BackgroundMethod::MorphOpen: {
            if (!validMask.empty()) return false;
            cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(k,k));
            cv::morphologyEx(image32f, computedBackground,
                             cv::MORPH_OPEN, kernel);
            break;
        }
        default:
            return false;
        }
    } catch (const cv::Exception&) {
        return false;
    } catch (const std::bad_alloc&) {
        return false;
    }
    background32f = std::move(computedBackground);
    return true;
}

}  // namespace singledie
