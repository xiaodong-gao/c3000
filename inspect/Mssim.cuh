#pragma once

#include "cuda_runtime.h"

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MssimCudaResult {
    double mssim;
} MssimCudaResult;

cudaError_t mssimCompute(const unsigned char* image1, const unsigned char* image2, int width,
                         int height, int channels, float* mapOut, MssimCudaResult* result);

cudaError_t mssimFuse(const unsigned char* templateImg, const unsigned char* alignedImg,
                      const float* ssimMap, int width, int height, int channels,
                      unsigned char* fusedOut);

void mssimReleaseWorkspace(void);

#ifdef __cplusplus
}

struct MssimDeviceView {
    const float* ssimMap = nullptr;
    const double* mssimDevice = nullptr;
    const unsigned char* fusion = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    const void* owner = nullptr;
    unsigned long long generation = 0;
};

class MssimCuda {
public:
    MssimCuda();
    ~MssimCuda();

    MssimCuda(const MssimCuda&) = delete;
    MssimCuda& operator=(const MssimCuda&) = delete;

    cudaError_t compute(const unsigned char* image1, const unsigned char* image2, int width,
                        int height, int channels, float* mapOut, double& mssim) const;

    cudaError_t fuse(const unsigned char* templateImg, const unsigned char* alignedImg,
                     const float* ssimMap, int width, int height, int channels,
                     unsigned char* fusedOut) const;

    // Device-resident path. image1/image2 must be device pointers. The borrowed view remains
    // valid until the next compute/host-fuse call, workspace resize, or engine destruction.
    // fuseResident and downloadResident do not invalidate it. Calls on one engine must be
    // serialized.
    cudaError_t computeResident(const unsigned char* image1Device,
                                 const unsigned char* image2Device, int width, int height,
                                 int channels, MssimDeviceView& view) const;

    // Uses the device SSIM map in view and leaves fusion in the internal workspace.
    cudaError_t fuseResident(const unsigned char* templateDevice,
                             const unsigned char* alignedDevice,
                             MssimDeviceView& view) const;

    cudaError_t downloadResident(const MssimDeviceView& view, float* mapOut,
                                  unsigned char* fusedOut, double& mssim) const;

    // Currently reserved device memory, useful for capacity planning and benchmarks.
    size_t workspaceBytes() const;

private:
    mutable void* deviceWorkspace_;
};

#endif
