#pragma once

#include "cuda_runtime.h"

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IntensityNormalizeCudaStats {
    int channels;
    double templateMean[4];
    double templateStdDev[4];
    double dataMean[4];
    double dataStdDev[4];
    double gain[4];
    double bias[4];
} IntensityNormalizeCudaStats;

// Host-pointer API. All input and output buffers are tightly packed.
cudaError_t intensityNormalizeCompute(
    const unsigned char* templateRoi, const unsigned char* dataRoi, int width, int height,
    int channels, unsigned char* templateNormalized, unsigned char* dataNormalized,
    unsigned char* absoluteDifference, float* l2Difference,
    IntensityNormalizeCudaStats* stats);

void intensityNormalizeReleaseWorkspace(void);

#ifdef __cplusplus
}

#include "IntensityNormalize.h"

struct IntensityNormalizeDeviceView {
    const unsigned char* templateRoi = nullptr;
    const unsigned char* dataNormalized = nullptr;
    const unsigned char* absoluteDifference = nullptr;
    const float* l2Difference = nullptr;
    const double* statistics = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    const void* owner = nullptr;
    unsigned long long generation = 0;
};

class IntensityNormalizeCuda {
public:
    IntensityNormalizeCuda();
    ~IntensityNormalizeCuda();

    IntensityNormalizeCuda(const IntensityNormalizeCuda&) = delete;
    IntensityNormalizeCuda& operator=(const IntensityNormalizeCuda&) = delete;

    // Reuses an internal workspace. Calls on one instance must be serialized; create one
    // engine (and PipelineContext) per concurrent wafer worker.
    cudaError_t compute(const unsigned char* templateRoi, const unsigned char* dataRoi,
                        int width, int height, int channels,
                        unsigned char* templateNormalized, unsigned char* dataNormalized,
                        unsigned char* absoluteDifference, float* l2Difference,
                        IntensityNormalizeStats& stats) const;

    // Uploads once and leaves all image results in the reusable device workspace. The borrowed
    // view remains valid until the next compute call, workspace resize, or engine destruction.
    // downloadResident does not invalidate it. Calls on one engine must be serialized.
    cudaError_t computeResident(const unsigned char* templateRoi, const unsigned char* dataRoi,
                                int width, int height, int channels,
                                IntensityNormalizeDeviceView& view) const;

    // Downloads the mutable outputs from a view produced by computeResident. The template is
    // unchanged and should be copied directly from the original host ROI.
    cudaError_t downloadResident(const IntensityNormalizeDeviceView& view,
                                 unsigned char* dataNormalized,
                                 unsigned char* absoluteDifference, float* l2Difference,
                                 IntensityNormalizeStats& stats) const;

    // Currently reserved device memory, useful for capacity planning and benchmarks.
    size_t workspaceBytes() const;

private:
    mutable void* deviceWorkspace_;
};

#endif
