#pragma once

// Histogram threshold for Die-to-Die binarization.
// CPU and CUDA share histogramThresholdComputeFromHistogram (smooth / peaks / T).
// CUDA only replaces rgbToGray, histogram counts, and apply.

#define HISTOGRAM_THRESHOLD_BINS 256
#define HISTOGRAM_THRESHOLD_MAX_PEAKS 64

struct HistogramPeak {
    int gray;
    float height;
};

struct HistogramThresholdParams {
    float gaussianSigma;
    float peakHeightRatio;
    int minPeakDistance;
    float minPeakCountRatio;
    int defaultThreshold;
};

struct HistogramThresholdResult {
    int threshold;
    int peakCount;
    int rawPeakCount;
    HistogramPeak rawPeaks[HISTOGRAM_THRESHOLD_MAX_PEAKS];
    HistogramPeak filteredPeaks[HISTOGRAM_THRESHOLD_MAX_PEAKS];
    char rule[80];
    float smoothed[HISTOGRAM_THRESHOLD_BINS];
    unsigned int histogram[HISTOGRAM_THRESHOLD_BINS];
};

HistogramThresholdParams histogramThresholdDefaultParams();
int histogramThresholdFuse(int templateT, int dataT);

// Host-side peak/threshold rules. Used by both CPU and CUDA after the histogram
// counts are available.
void histogramThresholdComputeFromHistogram(const unsigned int histogram[HISTOGRAM_THRESHOLD_BINS],
                                            int pixelCount, const HistogramThresholdParams& params,
                                            HistogramThresholdResult* result);

int histogramThresholdRgbToGrayCpu(const unsigned char* bgr, int nPixels, unsigned char* gray);
int histogramThresholdComputeHistogramCpu(const unsigned char* gray, int nPixels,
                                          unsigned int histogram[HISTOGRAM_THRESHOLD_BINS]);
int histogramThresholdComputeFromGrayCpu(const unsigned char* gray, int width, int height,
                                         const HistogramThresholdParams* params,
                                         HistogramThresholdResult* result);
int histogramThresholdComputeFromBgrCpu(const unsigned char* bgr, int width, int height,
                                        const HistogramThresholdParams* params,
                                        HistogramThresholdResult* result, unsigned char* grayOut);
int histogramThresholdApplyCpu(const unsigned char* gray, int nPixels, int threshold,
                               unsigned char* binaryOut);

class HistogramThresholdCpu {
public:
    HistogramThresholdCpu();
    explicit HistogramThresholdCpu(const HistogramThresholdParams& params);

    const HistogramThresholdParams& params() const { return params_; }
    void setParams(const HistogramThresholdParams& params) { params_ = params; }

    bool rgbToGray(const unsigned char* bgr, int nPixels, unsigned char* gray) const;
    bool computeHistogram(const unsigned char* gray, int nPixels,
                          unsigned int histogram[HISTOGRAM_THRESHOLD_BINS]) const;
    bool computeFromGray(const unsigned char* gray, int width, int height,
                         HistogramThresholdResult& result) const;
    bool computeFromBgr(const unsigned char* bgr, int width, int height,
                        HistogramThresholdResult& result, unsigned char* grayOut = nullptr) const;
    bool apply(const unsigned char* gray, int nPixels, int threshold,
               unsigned char* binaryOut) const;

    static int fuse(int templateT, int dataT);

private:
    HistogramThresholdParams params_;
};
