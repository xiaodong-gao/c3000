#pragma once

#include <opencv2/core.hpp>

class MssimCuda;

struct MssimResult {
    cv::Mat map;
    double mssim = 0.0;
};

// Gray (CV_8UC1) or color (CV_8UC3). Color MSSIM is the mean of B/G/R channels.
MssimResult computeMssim(const cv::Mat& image1, const cv::Mat& image2,
                         const cv::Mat& mask = cv::Mat());

MssimResult computeMssimCuda(MssimCuda& engine, const cv::Mat& image1, const cv::Mat& image2);
cv::Mat fuseByMssimCuda(MssimCuda& engine, const cv::Mat& templateImg, const cv::Mat& alignedImg,
                        const cv::Mat& ssimMap);

// SSIM-weighted fusion: out = template * ssim + aligned * (1 - ssim).
cv::Mat fuseByMssim(const cv::Mat& templateImg, const cv::Mat& alignedImg, const cv::Mat& ssimMap);

cv::Mat ssimMapToGray(const cv::Mat& ssimMap);
cv::Mat ssimMapToColor(const cv::Mat& ssimMap);
