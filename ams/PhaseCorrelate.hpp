#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// CPU phase correlation, matching OpenCV samples/cpp/phase_corr.cpp:
// convert to CV_64F, cache a Hanning window, then call cv::phaseCorrelate.
class PhaseCorrelate {
public:
    // Translation of dataImg relative to templateImg: +x right, +y down.
    // shiftImage(dataImg, shift) aligns data to the template.
    // useHanning=true 时走 cv::createHanningWindow，与 samples/cpp/phase_corr.cpp 一致。
    cv::Point2d compute(const cv::Mat& templateImg, const cv::Mat& dataImg,
                        double* response = nullptr, bool useHanning = true) const;

private:
    mutable cv::Mat hann_;
};

cv::Rect commonOverlapRoi(cv::Size size, const cv::Point2d& shift);

cv::Mat shiftImage(const cv::Mat& src, const cv::Point2d& shift,
                   int interpolation = cv::INTER_NEAREST);
