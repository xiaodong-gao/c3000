#include "PhaseCorrelate.hpp"

#include <algorithm>
#include <cmath>

cv::Point2d PhaseCorrelate::compute(const cv::Mat& templateImg, const cv::Mat& dataImg,
                                    double* response, bool useHanning) const {
    CV_Assert(!templateImg.empty() && !dataImg.empty());
    CV_Assert(templateImg.size() == dataImg.size());
    CV_Assert(templateImg.channels() == 1 && dataImg.channels() == 1);

    cv::Mat templateF;
    cv::Mat dataF;
    templateImg.convertTo(templateF, CV_64F);
    dataImg.convertTo(dataF, CV_64F);

    if (!useHanning) {
        return cv::phaseCorrelate(templateF, dataF, cv::noArray(), response);
    }
    if (hann_.size() != templateF.size() || hann_.type() != CV_64F) {
        cv::createHanningWindow(hann_, templateF.size(), CV_64F);
    }
    return cv::phaseCorrelate(templateF, dataF, hann_, response);
}

cv::Rect commonOverlapRoi(cv::Size size, const cv::Point2d& shift) {
    const double w = static_cast<double>(size.width);
    const double h = static_cast<double>(size.height);
    const int x0 = static_cast<int>(std::ceil(std::max(0.0, -shift.x)));
    const int y0 = static_cast<int>(std::ceil(std::max(0.0, -shift.y)));
    const int x1 = static_cast<int>(std::floor(std::min(w, w - shift.x)));
    const int y1 = static_cast<int>(std::floor(std::min(h, h - shift.y)));
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

cv::Mat shiftImage(const cv::Mat& src, const cv::Point2d& shift, int interpolation) {
    cv::Mat affine = cv::Mat::zeros(2, 3, CV_64F);
    affine.at<double>(0, 0) = 1.0;
    affine.at<double>(0, 2) = -shift.x;
    affine.at<double>(1, 1) = 1.0;
    affine.at<double>(1, 2) = -shift.y;
    cv::Mat dst;
    cv::warpAffine(src, dst, affine, src.size(), interpolation, cv::BORDER_CONSTANT, cv::Scalar(0));
    return dst;
}
