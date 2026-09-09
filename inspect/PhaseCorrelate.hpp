#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// CPU / OpenCV reference implementation. Prefer PhaseCorrelate.cuh for the
// high-precision CUDA version.
cv::Point2d phaseShiftBinary(const cv::Mat& templateBin, const cv::Mat& dataBin,
                             double* response = nullptr);

cv::Rect commonOverlapRoi(cv::Size size, const cv::Point2d& shift);

cv::Mat shiftImage(const cv::Mat& src, const cv::Point2d& shift,
                   int interpolation = cv::INTER_NEAREST);
