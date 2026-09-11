#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <string>

struct EccAlignParams {
    int maxIters = 100;
    double eps = 1e-5;
    int gaussFiltSize = 5;
    // OpenCV findTransformECC 运动模型，与 samples/cpp/image_alignment.cpp 的 -m 一致：
    // MOTION_TRANSLATION / MOTION_EUCLIDEAN / MOTION_AFFINE / MOTION_HOMOGRAPHY
    int motionType = cv::MOTION_EUCLIDEAN;
};

struct EccAlignResult {
    cv::Point2d shift;
    double cc = 0.0;
    bool ok = false;
    std::string error;
};

// Estimate the translation of dataImg relative to templateImg (same convention as
// PhaseCorrelate: +x right, +y down). Both images must be CV_8UC1 and same size.
EccAlignResult eccShift(const cv::Mat& templateImg, const cv::Mat& dataImg,
                        const EccAlignParams& params = {});
