#pragma once

#include "HistogramThreshold.h"
#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t histogramThresholdRgbToGray(const unsigned char* bgr, int nPixels,
                                        unsigned char* gray);

cudaError_t histogramThresholdComputeHistogram(const unsigned char* gray, int nPixels,
                                               unsigned int histogram[HISTOGRAM_THRESHOLD_BINS]);

cudaError_t histogramThresholdComputeFromBgr(const unsigned char* bgr, int width, int height,
                                             const HistogramThresholdParams* params,
                                             HistogramThresholdResult* result,
                                             unsigned char* grayOut);

cudaError_t histogramThresholdComputeFromGray(const unsigned char* gray, int width, int height,
                                              const HistogramThresholdParams* params,
                                              HistogramThresholdResult* result);

cudaError_t histogramThresholdApply(const unsigned char* gray, int nPixels, int threshold,
                                    unsigned char* binaryOut);

void histogramThresholdReleaseWorkspace(void);

#ifdef __cplusplus
}

class HistogramThreshold {
public:
    HistogramThreshold();
    explicit HistogramThreshold(const HistogramThresholdParams& params);
    ~HistogramThreshold();

    HistogramThreshold(const HistogramThreshold&) = delete;
    HistogramThreshold& operator=(const HistogramThreshold&) = delete;

    const HistogramThresholdParams& params() const { return params_; }
    void setParams(const HistogramThresholdParams& params) { params_ = params; }

    cudaError_t rgbToGray(const unsigned char* bgr, int nPixels, unsigned char* gray) const;
    cudaError_t computeHistogram(const unsigned char* gray, int nPixels,
                                 unsigned int histogram[HISTOGRAM_THRESHOLD_BINS]) const;
    cudaError_t computeFromBgr(const unsigned char* bgr, int width, int height,
                               HistogramThresholdResult& result,
                               unsigned char* grayOut = nullptr) const;
    cudaError_t computeFromGray(const unsigned char* gray, int width, int height,
                                HistogramThresholdResult& result) const;
    cudaError_t apply(const unsigned char* gray, int nPixels, int threshold,
                      unsigned char* binaryOut) const;

    bool rgbToGrayCpu(const unsigned char* bgr, int nPixels, unsigned char* gray) const;
    bool computeFromGrayCpu(const unsigned char* gray, int width, int height,
                            HistogramThresholdResult& result) const;
    bool computeFromBgrCpu(const unsigned char* bgr, int width, int height,
                           HistogramThresholdResult& result, unsigned char* grayOut = nullptr) const;
    bool applyCpu(const unsigned char* gray, int nPixels, int threshold,
                  unsigned char* binaryOut) const;

    static int fuse(int templateT, int dataT);

private:
    HistogramThresholdParams params_;
    mutable void* deviceWorkspace_;
};

#endif
