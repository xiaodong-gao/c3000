#include "PhaseCorrelate.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <cufft.h>

namespace phase_corr_detail {

constexpr int kBlockSize = 256;
constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-15;
constexpr int kPatchRadius = 2;

#define CUDA_CHECK(call)                 \
    do {                                 \
        const cudaError_t _err = (call); \
        if (_err != cudaSuccess) {       \
            return _err;                 \
        }                                \
    } while (0)

#define CUFFT_CHECK(call)                            \
    do {                                             \
        if ((call) != CUFFT_SUCCESS) {               \
            return cudaErrorUnknown;                 \
        }                                            \
    } while (0)

struct DeviceWorkspace {
    unsigned char* dTmplU8 = nullptr;
    unsigned char* dDataU8 = nullptr;
    double* dTmpl = nullptr;
    double* dData = nullptr;
    cufftDoubleComplex* dFft1 = nullptr;
    cufftDoubleComplex* dFft2 = nullptr;
    double* dCorr = nullptr;
    double* dPatch = nullptr;
    cufftHandle planFwd = 0;
    cufftHandle planInv = 0;
    int width = 0;
    int height = 0;
    size_t u8Cap = 0;
    size_t realCap = 0;
    size_t specCap = 0;

    cudaError_t ensureBytes(void** ptr, size_t* cap, size_t bytes) {
        if (*cap >= bytes) {
            return cudaSuccess;
        }
        if (*ptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
            *cap = 0;
        }
        const cudaError_t status = cudaMalloc(ptr, bytes);
        if (status == cudaSuccess) {
            *cap = bytes;
        }
        return status;
    }

    cudaError_t ensure(int w, int h) {
        const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t specN = static_cast<size_t>(h) * static_cast<size_t>(w / 2 + 1);
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dTmplU8), &u8Cap, n * 2));
        dDataU8 = dTmplU8 + n;
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dTmpl), &realCap, n * sizeof(double) * 3));
        dData = dTmpl + n;
        dCorr = dTmpl + 2 * n;
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dFft1), &specCap,
                               specN * sizeof(cufftDoubleComplex) * 2));
        dFft2 = dFft1 + specN;
        if (!dPatch) {
            CUDA_CHECK(cudaMalloc(&dPatch, (2 * kPatchRadius + 1) * (2 * kPatchRadius + 1) *
                                               sizeof(double)));
        }

        if (width != w || height != h || planFwd == 0 || planInv == 0) {
            if (planFwd) {
                cufftDestroy(planFwd);
                planFwd = 0;
            }
            if (planInv) {
                cufftDestroy(planInv);
                planInv = 0;
            }
            CUFFT_CHECK(cufftPlan2d(&planFwd, h, w, CUFFT_D2Z));
            CUFFT_CHECK(cufftPlan2d(&planInv, h, w, CUFFT_Z2D));
            width = w;
            height = h;
        }
        return cudaSuccess;
    }

    void release() {
        if (planFwd) {
            cufftDestroy(planFwd);
            planFwd = 0;
        }
        if (planInv) {
            cufftDestroy(planInv);
            planInv = 0;
        }
        cudaFree(dTmplU8);
        cudaFree(dTmpl);
        cudaFree(dFft1);
        cudaFree(dPatch);
        dTmplU8 = dDataU8 = nullptr;
        dTmpl = dData = dCorr = dPatch = nullptr;
        dFft1 = dFft2 = nullptr;
        u8Cap = realCap = specCap = 0;
        width = height = 0;
    }
};

DeviceWorkspace& defaultWorkspace() {
    static DeviceWorkspace workspace;
    return workspace;
}

DeviceWorkspace& workspaceFrom(void* ptr) {
    return ptr ? *static_cast<DeviceWorkspace*>(ptr) : defaultWorkspace();
}

__global__ void windowedDoubleKernel(const unsigned char* __restrict__ src, double* __restrict__ dst,
                                     int width, int height) {
    const int n = width * height;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % width;
        const int y = i / width;
        const double wx =
            width > 1 ? 0.5 * (1.0 - cos(2.0 * kPi * static_cast<double>(x) / (width - 1))) : 1.0;
        const double wy =
            height > 1 ? 0.5 * (1.0 - cos(2.0 * kPi * static_cast<double>(y) / (height - 1))) : 1.0;
        dst[i] = static_cast<double>(src[i]) * wx * wy;
    }
}

__global__ void crossPowerKernel(const cufftDoubleComplex* __restrict__ fft1,
                                 const cufftDoubleComplex* __restrict__ fft2,
                                 cufftDoubleComplex* __restrict__ cross, int specN) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < specN; i += blockDim.x * gridDim.x) {
        const cufftDoubleComplex a = fft1[i];
        const cufftDoubleComplex b = fft2[i];
        const double re = a.x * b.x + a.y * b.y;
        const double im = a.y * b.x - a.x * b.y;
        const double mag = hypot(re, im) + kEps;
        cufftDoubleComplex o;
        o.x = re / mag;
        o.y = im / mag;
        cross[i] = o;
    }
}

__global__ void scaleCorrKernel(double* __restrict__ corr, int n, double scale) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        corr[i] *= scale;
    }
}

__global__ void gatherPatchKernel(const double* __restrict__ corr, int width, int height, int cx,
                                  int cy, int radius, double* __restrict__ patch) {
    const int dim = 2 * radius + 1;
    const int n = dim * dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int dx = (i % dim) - radius;
        const int dy = (i / dim) - radius;
        const int x = ((cx + dx) % width + width) % width;
        const int y = ((cy + dy) % height + height) % height;
        patch[i] = corr[y * width + x];
    }
}

int launchGrid(int n) {
    const int grid = (n + kBlockSize - 1) / kBlockSize;
    return grid < 1 ? 1 : grid;
}

int maxShiftFor(int width, int height) {
    int maxShift = std::max(16, std::min(48, std::min(width, height) / 8));
    maxShift = std::min(maxShift, std::min(width, height));
    return std::max(1, maxShift);
}

void wrapShift(double* sx, double* sy, int width, int height) {
    if (*sx > width * 0.5) {
        *sx -= width;
    }
    if (*sy > height * 0.5) {
        *sy -= height;
    }
}

cudaError_t subpixelPeak(DeviceWorkspace& ws, int peakX, int peakY, double* outX, double* outY) {
    const int dim = 2 * kPatchRadius + 1;
    gatherPatchKernel<<<1, dim * dim>>>(ws.dCorr, ws.width, ws.height, peakX, peakY, kPatchRadius,
                                        ws.dPatch);
    CUDA_CHECK(cudaGetLastError());

    double patch[2 * kPatchRadius + 1][2 * kPatchRadius + 1];
    CUDA_CHECK(cudaMemcpy(patch, ws.dPatch, sizeof(patch), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    for (int dy = -kPatchRadius; dy <= kPatchRadius; ++dy) {
        for (int dx = -kPatchRadius; dx <= kPatchRadius; ++dx) {
            const double v = std::max(0.0, patch[dy + kPatchRadius][dx + kPatchRadius]);
            sum += v;
            sx += v * static_cast<double>(peakX + dx);
            sy += v * static_cast<double>(peakY + dy);
        }
    }
    if (sum <= kEps) {
        *outX = static_cast<double>(peakX);
        *outY = static_cast<double>(peakY);
        return cudaSuccess;
    }
    *outX = sx / sum;
    *outY = sy / sum;
    return cudaSuccess;
}

cudaError_t makeCandidate(DeviceWorkspace& ws, int x0, int y0, int maxShift, double* bestMag,
                          double* bestValue, double* bestDx, double* bestDy) {
    std::vector<double> roi(static_cast<size_t>(maxShift) * maxShift);
    CUDA_CHECK(cudaMemcpy2D(roi.data(), maxShift * sizeof(double),
                            ws.dCorr + static_cast<size_t>(y0) * ws.width + x0,
                            ws.width * sizeof(double), maxShift * sizeof(double), maxShift,
                            cudaMemcpyDeviceToHost));

    int best = 0;
    double value = roi[0];
    for (int i = 1; i < maxShift * maxShift; ++i) {
        if (roi[i] > value) {
            value = roi[i];
            best = i;
        }
    }

    const int peakX = x0 + best % maxShift;
    const int peakY = y0 + best / maxShift;
    double subX = 0.0;
    double subY = 0.0;
    CUDA_CHECK(subpixelPeak(ws, peakX, peakY, &subX, &subY));

    wrapShift(&subX, &subY, ws.width, ws.height);
    const double dx = -subX;
    const double dy = -subY;
    const double mag = std::hypot(dx, dy);
    if (value > *bestValue + 1e-12 ||
        (std::abs(value - *bestValue) <= 1e-12 && mag < *bestMag)) {
        *bestMag = mag;
        *bestValue = value;
        *bestDx = dx;
        *bestDy = dy;
    }
    return cudaSuccess;
}

cudaError_t computeDevice(DeviceWorkspace& ws, const unsigned char* templateImg,
                          const unsigned char* dataImg, int width, int height,
                          PhaseCorrelateResult* result) {
    if (!templateImg || !dataImg || !result || width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }
    memset(result, 0, sizeof(*result));
    CUDA_CHECK(ws.ensure(width, height));

    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    const int specN = height * (width / 2 + 1);
    CUDA_CHECK(cudaMemcpy(ws.dTmplU8, templateImg, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.dDataU8, dataImg, n, cudaMemcpyHostToDevice));

    windowedDoubleKernel<<<launchGrid(static_cast<int>(n)), kBlockSize>>>(ws.dTmplU8, ws.dTmpl,
                                                                          width, height);
    windowedDoubleKernel<<<launchGrid(static_cast<int>(n)), kBlockSize>>>(ws.dDataU8, ws.dData,
                                                                          width, height);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecD2Z(ws.planFwd, ws.dTmpl, ws.dFft1));
    CUFFT_CHECK(cufftExecD2Z(ws.planFwd, ws.dData, ws.dFft2));
    crossPowerKernel<<<launchGrid(specN), kBlockSize>>>(ws.dFft1, ws.dFft2, ws.dFft1, specN);
    CUDA_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftExecZ2D(ws.planInv, ws.dFft1, ws.dCorr));
    scaleCorrKernel<<<launchGrid(static_cast<int>(n)), kBlockSize>>>(
        ws.dCorr, static_cast<int>(n), 1.0 / static_cast<double>(n));
    CUDA_CHECK(cudaGetLastError());

    const int maxShift = maxShiftFor(width, height);
    double bestMag = 1e300;
    double bestValue = -1e300;
    double bestDx = 0.0;
    double bestDy = 0.0;
    CUDA_CHECK(makeCandidate(ws, 0, 0, maxShift, &bestMag, &bestValue, &bestDx, &bestDy));
    CUDA_CHECK(makeCandidate(ws, width - maxShift, 0, maxShift, &bestMag, &bestValue, &bestDx,
                             &bestDy));
    CUDA_CHECK(makeCandidate(ws, 0, height - maxShift, maxShift, &bestMag, &bestValue, &bestDx,
                             &bestDy));
    CUDA_CHECK(makeCandidate(ws, width - maxShift, height - maxShift, maxShift, &bestMag,
                             &bestValue, &bestDx, &bestDy));

    result->dx = bestDx;
    result->dy = bestDy;
    result->response = bestValue;
    return cudaSuccess;
}

#undef CUDA_CHECK
#undef CUFFT_CHECK

}  // namespace phase_corr_detail

using phase_corr_detail::DeviceWorkspace;
using phase_corr_detail::computeDevice;
using phase_corr_detail::defaultWorkspace;
using phase_corr_detail::workspaceFrom;

cudaError_t phaseCorrelateCompute(const unsigned char* templateImg, const unsigned char* dataImg,
                                  int width, int height, PhaseCorrelateResult* result) {
    return computeDevice(defaultWorkspace(), templateImg, dataImg, width, height, result);
}

void phaseCorrelateReleaseWorkspace(void) { defaultWorkspace().release(); }

PhaseCorrelateCuda::PhaseCorrelateCuda() : deviceWorkspace_(new DeviceWorkspace()) {}

PhaseCorrelateCuda::~PhaseCorrelateCuda() {
    if (deviceWorkspace_) {
        static_cast<DeviceWorkspace*>(deviceWorkspace_)->release();
        delete static_cast<DeviceWorkspace*>(deviceWorkspace_);
        deviceWorkspace_ = nullptr;
    }
}

cudaError_t PhaseCorrelateCuda::compute(const unsigned char* templateImg,
                                        const unsigned char* dataImg, int width, int height,
                                        PhaseCorrelateResult& result) const {
    return computeDevice(workspaceFrom(deviceWorkspace_), templateImg, dataImg, width, height,
                         &result);
}
