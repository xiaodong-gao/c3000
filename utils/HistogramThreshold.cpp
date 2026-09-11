#include "HistogramThreshold.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kMaxKernelRadius = 31;

struct Peak {
    int gray;
    float height;
};

void setRule(HistogramThresholdResult* result, const char* text) {
#ifdef _MSC_VER
    strncpy_s(result->rule, text, _TRUNCATE);
#else
    std::snprintf(result->rule, sizeof(result->rule), "%s", text);
#endif
}

void gaussianSmooth(const float* input, float* output, int n, float sigma) {
    const int radius =
        std::max(1, std::min(kMaxKernelRadius, static_cast<int>(std::ceil(3.0f * sigma))));
    const int kernelSize = 2 * radius + 1;
    float kernel[2 * kMaxKernelRadius + 1];
    const float twoSigma2 = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float v = std::exp(-(static_cast<float>(i * i)) / twoSigma2);
        kernel[i + radius] = v;
        sum += v;
    }
    const float invSum = 1.0f / sum;
    for (int i = 0; i < kernelSize; ++i) {
        kernel[i] *= invSum;
    }

    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int j = i + k;
            if (j < 0) {
                j = -j;
            } else if (j >= n) {
                j = 2 * n - 2 - j;
            }
            acc += input[j] * kernel[k + radius];
        }
        output[i] = acc;
    }
}

int findPeaks(const float* hist, Peak* peaks, int maxPeaks) {
    int count = 0;
    auto push = [&](int gray, float height) {
        if (count < maxPeaks) {
            peaks[count++] = {gray, height};
        }
    };

    if (hist[0] > hist[1]) {
        push(0, hist[0]);
    }

    int i = 1;
    while (i < HISTOGRAM_THRESHOLD_BINS - 1) {
        if (hist[i] > hist[i - 1]) {
            int j = i;
            while (j < HISTOGRAM_THRESHOLD_BINS - 1 && hist[j] == hist[j + 1]) {
                ++j;
            }
            if (j < HISTOGRAM_THRESHOLD_BINS - 1 && hist[j] > hist[j + 1]) {
                const int center = (i + j) / 2;
                push(center, hist[center]);
            }
            i = j + 1;
        } else {
            ++i;
        }
    }

    if (hist[HISTOGRAM_THRESHOLD_BINS - 1] > hist[HISTOGRAM_THRESHOLD_BINS - 2]) {
        push(HISTOGRAM_THRESHOLD_BINS - 1, hist[HISTOGRAM_THRESHOLD_BINS - 1]);
    }
    return count;
}

int filterPeaks(const Peak* peaks, int peakCount, int pixelCount,
                const HistogramThresholdParams& params, Peak* selected) {
    if (peakCount <= 0) {
        return 0;
    }

    float maxHeight = peaks[0].height;
    for (int i = 1; i < peakCount; ++i) {
        maxHeight = std::max(maxHeight, peaks[i].height);
    }

    const float heightGate = maxHeight * params.peakHeightRatio;
    const float countGate = static_cast<float>(pixelCount) * params.minPeakCountRatio;

    Peak candidates[HISTOGRAM_THRESHOLD_MAX_PEAKS];
    int candidateCount = 0;
    for (int i = 0; i < peakCount; ++i) {
        if (peaks[i].height >= heightGate && peaks[i].height >= countGate &&
            candidateCount < HISTOGRAM_THRESHOLD_MAX_PEAKS) {
            candidates[candidateCount++] = peaks[i];
        }
    }

    std::sort(candidates, candidates + candidateCount,
              [](const Peak& a, const Peak& b) { return a.height > b.height; });

    int selectedCount = 0;
    for (int i = 0; i < candidateCount; ++i) {
        bool farEnough = true;
        for (int j = 0; j < selectedCount; ++j) {
            if (std::abs(candidates[i].gray - selected[j].gray) < params.minPeakDistance) {
                farEnough = false;
                break;
            }
        }
        if (farEnough) {
            selected[selectedCount++] = candidates[i];
        }
    }

    std::sort(selected, selected + selectedCount,
              [](const Peak& a, const Peak& b) { return a.gray < b.gray; });
    return selectedCount;
}

void copyPeaks(const Peak* src, int srcCount, HistogramPeak* dst, int* count) {
    const int n = srcCount < HISTOGRAM_THRESHOLD_MAX_PEAKS ? srcCount : HISTOGRAM_THRESHOLD_MAX_PEAKS;
    *count = n;
    for (int i = 0; i < n; ++i) {
        dst[i].gray = src[i].gray;
        dst[i].height = src[i].height;
    }
}

void rgbToGrayHost(const unsigned char* bgr, int nPixels, unsigned char* gray) {
    for (int i = 0; i < nPixels; ++i) {
        const unsigned char b = bgr[i * 3 + 0];
        const unsigned char g = bgr[i * 3 + 1];
        const unsigned char r = bgr[i * 3 + 2];
        gray[i] = static_cast<unsigned char>((77 * r + 150 * g + 29 * b) >> 8);
    }
}

void histogramHost(const unsigned char* gray, int nPixels, unsigned int* hist) {
    std::memset(hist, 0, HISTOGRAM_THRESHOLD_BINS * sizeof(unsigned int));
    for (int i = 0; i < nPixels; ++i) {
        ++hist[gray[i]];
    }
}

void applyHost(const unsigned char* gray, int nPixels, int threshold, unsigned char* binaryOut) {
    for (int i = 0; i < nPixels; ++i) {
        binaryOut[i] = gray[i] > threshold ? 255 : 0;
    }
}

}  // namespace

HistogramThresholdParams histogramThresholdDefaultParams() {
    HistogramThresholdParams params;
    params.gaussianSigma = 2.5f;
    params.peakHeightRatio = 0.05f;
    params.minPeakDistance = 15;
    params.minPeakCountRatio = 0.005f;
    params.defaultThreshold = 125;
    return params;
}

int histogramThresholdFuse(int templateT, int dataT) { return (templateT + dataT) / 2; }

void histogramThresholdComputeFromHistogram(const unsigned int histogram[HISTOGRAM_THRESHOLD_BINS],
                                            int pixelCount, const HistogramThresholdParams& params,
                                            HistogramThresholdResult* result) {
    if (!histogram || !result) {
        return;
    }

    float rawHist[HISTOGRAM_THRESHOLD_BINS];
    for (int i = 0; i < HISTOGRAM_THRESHOLD_BINS; ++i) {
        rawHist[i] = static_cast<float>(histogram[i]);
        result->histogram[i] = histogram[i];
    }

    gaussianSmooth(rawHist, result->smoothed, HISTOGRAM_THRESHOLD_BINS, params.gaussianSigma);

    Peak rawPeaks[HISTOGRAM_THRESHOLD_MAX_PEAKS];
    Peak filtered[HISTOGRAM_THRESHOLD_MAX_PEAKS];
    const int rawCount = findPeaks(result->smoothed, rawPeaks, HISTOGRAM_THRESHOLD_MAX_PEAKS);
    const int filteredCount = filterPeaks(rawPeaks, rawCount, pixelCount, params, filtered);
    copyPeaks(rawPeaks, rawCount, result->rawPeaks, &result->rawPeakCount);
    copyPeaks(filtered, filteredCount, result->filteredPeaks, &result->peakCount);

    if (result->peakCount == 0) {
        result->threshold = params.defaultThreshold;
        setRule(result, "0 peaks -> T=125");
    } else if (result->peakCount == 1) {
        result->threshold = (result->filteredPeaks[0].gray + 255) / 2;
        setRule(result, "1 peak -> T=(peak+255)/2");
    } else if (result->peakCount == 2) {
        result->threshold = (result->filteredPeaks[0].gray + result->filteredPeaks[1].gray) / 2;
        setRule(result, "2 peaks -> T=(peak1+peak2)/2");
    } else {
        int top0 = 0;
        int top1 = 1;
        if (filtered[1].height > filtered[0].height) {
            top0 = 1;
            top1 = 0;
        }
        for (int i = 2; i < result->peakCount; ++i) {
            if (filtered[i].height > filtered[top0].height) {
                top1 = top0;
                top0 = i;
            } else if (filtered[i].height > filtered[top1].height) {
                top1 = i;
            }
        }
        result->threshold = (filtered[top0].gray + filtered[top1].gray) / 2;
        setRule(result, ">2 peaks -> T=(top1+top2)/2");
    }
}

int histogramThresholdRgbToGrayCpu(const unsigned char* bgr, int nPixels, unsigned char* gray) {
    if (!bgr || !gray || nPixels <= 0) {
        return -1;
    }
    rgbToGrayHost(bgr, nPixels, gray);
    return 0;
}

int histogramThresholdComputeHistogramCpu(const unsigned char* gray, int nPixels,
                                          unsigned int histogram[HISTOGRAM_THRESHOLD_BINS]) {
    if (!gray || !histogram || nPixels <= 0) {
        return -1;
    }
    histogramHost(gray, nPixels, histogram);
    return 0;
}

int histogramThresholdComputeFromGrayCpu(const unsigned char* gray, int width, int height,
                                         const HistogramThresholdParams* params,
                                         HistogramThresholdResult* result) {
    if (!gray || !result || width <= 0 || height <= 0) {
        return -1;
    }
    std::memset(result, 0, sizeof(*result));
    const HistogramThresholdParams localParams =
        params ? *params : histogramThresholdDefaultParams();
    const int nPixels = width * height;
    unsigned int hist[HISTOGRAM_THRESHOLD_BINS];
    histogramHost(gray, nPixels, hist);
    histogramThresholdComputeFromHistogram(hist, nPixels, localParams, result);
    return 0;
}

int histogramThresholdComputeFromBgrCpu(const unsigned char* bgr, int width, int height,
                                        const HistogramThresholdParams* params,
                                        HistogramThresholdResult* result, unsigned char* grayOut) {
    if (!bgr || !result || width <= 0 || height <= 0) {
        return -1;
    }
    const int nPixels = width * height;
    std::vector<unsigned char> localGray;
    unsigned char* gray = grayOut;
    if (!gray) {
        localGray.resize(static_cast<size_t>(nPixels));
        gray = localGray.data();
    }
    rgbToGrayHost(bgr, nPixels, gray);
    return histogramThresholdComputeFromGrayCpu(gray, width, height, params, result);
}

int histogramThresholdApplyCpu(const unsigned char* gray, int nPixels, int threshold,
                               unsigned char* binaryOut) {
    if (!gray || !binaryOut || nPixels <= 0) {
        return -1;
    }
    applyHost(gray, nPixels, threshold, binaryOut);
    return 0;
}

HistogramThresholdCpu::HistogramThresholdCpu() : params_(histogramThresholdDefaultParams()) {}

HistogramThresholdCpu::HistogramThresholdCpu(const HistogramThresholdParams& params)
    : params_(params) {}

bool HistogramThresholdCpu::rgbToGray(const unsigned char* bgr, int nPixels,
                                      unsigned char* gray) const {
    return histogramThresholdRgbToGrayCpu(bgr, nPixels, gray) == 0;
}

bool HistogramThresholdCpu::computeHistogram(
    const unsigned char* gray, int nPixels, unsigned int histogram[HISTOGRAM_THRESHOLD_BINS]) const {
    return histogramThresholdComputeHistogramCpu(gray, nPixels, histogram) == 0;
}

bool HistogramThresholdCpu::computeFromGray(const unsigned char* gray, int width, int height,
                                            HistogramThresholdResult& result) const {
    return histogramThresholdComputeFromGrayCpu(gray, width, height, &params_, &result) == 0;
}

bool HistogramThresholdCpu::computeFromBgr(const unsigned char* bgr, int width, int height,
                                           HistogramThresholdResult& result,
                                           unsigned char* grayOut) const {
    return histogramThresholdComputeFromBgrCpu(bgr, width, height, &params_, &result, grayOut) == 0;
}

bool HistogramThresholdCpu::apply(const unsigned char* gray, int nPixels, int threshold,
                                  unsigned char* binaryOut) const {
    return histogramThresholdApplyCpu(gray, nPixels, threshold, binaryOut) == 0;
}

int HistogramThresholdCpu::fuse(int templateT, int dataT) {
    return histogramThresholdFuse(templateT, dataT);
}
