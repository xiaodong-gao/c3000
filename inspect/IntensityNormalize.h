#pragma once

#include <array>
#include <string>

#include <opencv2/core.hpp>

class IntensityNormalizeCuda;

struct IntensityNormalizeStats {
    int channels = 0;
    std::array<double, 4> templateMean{};
    std::array<double, 4> templateStdDev{};
    std::array<double, 4> dataMean{};
    std::array<double, 4> dataStdDev{};
    std::array<double, 4> gain{};
    std::array<double, 4> bias{};
};

struct IntensityNormalizeOutput {
    bool ok = false;
    std::string error;
    cv::Mat templateNormalized;  // Reference image; copied without radiometric modification.
    cv::Mat dataNormalized;      // data * gain + bias, independently for every channel.
    cv::Mat absoluteDifference;  // CV_8U, same channel count as the input.
    cv::Mat l2Difference;        // CV_32FC1, Euclidean distance across channels per pixel.
    IntensityNormalizeStats stats;
};

// Median + MAD inliers from a 256-bin 8-bit histogram. Shared by CPU and CUDA so both
// backends apply the same robust gain/bias and large defects do not dominate statistics.
void robustMeanStdFromU8Histogram(const unsigned int histogram[256], int pixelCount,
                                  double& meanOut, double& stdOut);

// Match dataRoi to templateRoi using robust (outlier-resistant) mean/std so a large
// defect does not dominate gain/bias. Supported formats: CV_8UC1 and CV_8UC3.
// If the data standard deviation is zero, gain=1 and only the mean offset is corrected.
IntensityNormalizeOutput intensityNormalize(const cv::Mat& templateRoi, const cv::Mat& dataRoi);

// Same robust mapping as intensityNormalize: GPU histograms + host robust stats + GPU
// per-pixel gain/bias, absolute difference, and L2.
IntensityNormalizeOutput intensityNormalizeCuda(IntensityNormalizeCuda& engine,
                                                const cv::Mat& templateRoi,
                                                const cv::Mat& dataRoi);
