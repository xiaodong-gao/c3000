#include "IntensityNormalize.h"

#include "IntensityNormalize.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

bool validatePair(const cv::Mat& templateRoi, const cv::Mat& dataRoi, std::string& error) {
    if (templateRoi.empty() || dataRoi.empty()) {
        error = "intensity normalization requires non-empty ROIs";
        return false;
    }
    if (templateRoi.size() != dataRoi.size() || templateRoi.type() != dataRoi.type()) {
        error = "intensity normalization requires same-size, same-type ROIs";
        return false;
    }
    if (templateRoi.type() != CV_8UC1 && templateRoi.type() != CV_8UC3) {
        error = "intensity normalization supports only CV_8UC1 and CV_8UC3";
        return false;
    }
    return true;
}

void computeDifferences(const cv::Mat& reference, const cv::Mat& normalized,
                        cv::Mat& absoluteDifference, cv::Mat& l2Difference) {
    cv::absdiff(reference, normalized, absoluteDifference);

    if (reference.channels() == 1) {
        absoluteDifference.convertTo(l2Difference, CV_32F);
        return;
    }

    cv::Mat referenceF;
    cv::Mat normalizedF;
    reference.convertTo(referenceF, CV_32F);
    normalized.convertTo(normalizedF, CV_32F);
    const cv::Mat delta = referenceF - normalizedF;
    std::vector<cv::Mat> channels;
    cv::split(delta.mul(delta), channels);
    l2Difference = channels[0] + channels[1] + channels[2];
    cv::sqrt(l2Difference, l2Difference);
}

void robustMeanStd(const cv::Mat& plane, double& meanOut, double& stdOut) {
    const cv::Mat src = plane.isContinuous() ? plane : plane.clone();
    unsigned int hist[256] = {};
    const int n = src.rows * src.cols;
    const unsigned char* p = src.ptr<unsigned char>(0);
    for (int i = 0; i < n; ++i) {
        ++hist[p[i]];
    }
    robustMeanStdFromU8Histogram(hist, n, meanOut, stdOut);
}

}  // namespace

void robustMeanStdFromU8Histogram(const unsigned int histogram[256], int pixelCount,
                                  double& meanOut, double& stdOut) {
    if (pixelCount <= 0) {
        meanOut = 0.0;
        stdOut = 0.0;
        return;
    }

    const int n = pixelCount;
    auto percentileValue = [&](double frac) {
        const int target = std::min(n - 1, std::max(0, static_cast<int>(frac * (n - 1))));
        long long acc = 0;
        for (int v = 0; v < 256; ++v) {
            acc += static_cast<long long>(histogram[v]);
            if (acc > target) {
                return v;
            }
        }
        return 255;
    };
    const int median = percentileValue(0.5);

    unsigned int distHist[256] = {};
    for (int v = 0; v < 256; ++v) {
        distHist[std::abs(v - median)] += histogram[v];
    }
    long long acc = 0;
    int mad = 0;
    const int madTarget = std::min(n - 1, std::max(0, (n - 1) / 2));
    for (int d = 0; d < 256; ++d) {
        acc += static_cast<long long>(distHist[d]);
        if (acc > madTarget) {
            mad = d;
            break;
        }
    }

    const double sigma = 1.4826 * static_cast<double>(mad);
    if (sigma < 1e-9) {
        meanOut = static_cast<double>(median);
        stdOut = 0.0;
        return;
    }

    const double gate = 3.0 * sigma;
    double sum = 0.0;
    double sumSq = 0.0;
    int count = 0;
    for (int v = 0; v < 256; ++v) {
        if (std::abs(v - median) > gate) {
            continue;
        }
        const int c = static_cast<int>(histogram[v]);
        sum += static_cast<double>(v) * c;
        sumSq += static_cast<double>(v) * v * c;
        count += c;
    }
    if (count <= 0) {
        meanOut = static_cast<double>(median);
        stdOut = sigma;
        return;
    }
    meanOut = sum / count;
    stdOut = std::sqrt(std::max(0.0, sumSq / count - meanOut * meanOut));
}

IntensityNormalizeOutput intensityNormalize(const cv::Mat& templateRoi, const cv::Mat& dataRoi) {
    IntensityNormalizeOutput out;
    if (!validatePair(templateRoi, dataRoi, out.error)) {
        return out;
    }

    const cv::Mat reference = templateRoi.isContinuous() ? templateRoi : templateRoi.clone();
    const cv::Mat source = dataRoi.isContinuous() ? dataRoi : dataRoi.clone();
    const int channels = reference.channels();
    out.stats.channels = channels;

    std::vector<cv::Mat> sourcePlanes;
    std::vector<cv::Mat> referencePlanes;
    cv::split(source, sourcePlanes);
    cv::split(reference, referencePlanes);
    std::vector<cv::Mat> normalizedPlanes;
    normalizedPlanes.resize(static_cast<size_t>(channels));

    constexpr double kFlatEpsilon = 1e-9;
    for (int c = 0; c < channels; ++c) {
        double templateMean = 0.0;
        double templateStdDev = 0.0;
        double dataMean = 0.0;
        double dataStdDev = 0.0;
        robustMeanStd(referencePlanes[c], templateMean, templateStdDev);
        robustMeanStd(sourcePlanes[c], dataMean, dataStdDev);
        const double gain = dataStdDev > kFlatEpsilon ? templateStdDev / dataStdDev : 1.0;
        const double bias = templateMean - gain * dataMean;
        sourcePlanes[c].convertTo(normalizedPlanes[c], CV_8U, gain, bias);

        out.stats.templateMean[c] = templateMean;
        out.stats.templateStdDev[c] = templateStdDev;
        out.stats.dataMean[c] = dataMean;
        out.stats.dataStdDev[c] = dataStdDev;
        out.stats.gain[c] = gain;
        out.stats.bias[c] = bias;
    }

    out.templateNormalized = reference.clone();
    cv::merge(normalizedPlanes, out.dataNormalized);
    computeDifferences(out.templateNormalized, out.dataNormalized, out.absoluteDifference,
                       out.l2Difference);
    out.ok = true;
    return out;
}

IntensityNormalizeOutput intensityNormalizeCuda(IntensityNormalizeCuda& engine,
                                                const cv::Mat& templateRoi,
                                                const cv::Mat& dataRoi) {
    IntensityNormalizeOutput out;
    if (!validatePair(templateRoi, dataRoi, out.error)) {
        return out;
    }

    const cv::Mat reference = templateRoi.isContinuous() ? templateRoi : templateRoi.clone();
    const cv::Mat source = dataRoi.isContinuous() ? dataRoi : dataRoi.clone();
    out.templateNormalized.create(reference.size(), reference.type());
    out.dataNormalized.create(reference.size(), reference.type());
    out.absoluteDifference.create(reference.size(), reference.type());
    out.l2Difference.create(reference.size(), CV_32FC1);

    const cudaError_t status = engine.compute(
        reference.data, source.data, reference.cols, reference.rows, reference.channels(),
        out.templateNormalized.data, out.dataNormalized.data, out.absoluteDifference.data,
        out.l2Difference.ptr<float>(), out.stats);
    if (status != cudaSuccess) {
        out.error = cudaGetErrorString(status);
        return out;
    }
    out.ok = true;
    return out;
}
