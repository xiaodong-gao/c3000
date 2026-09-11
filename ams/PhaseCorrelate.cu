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
constexpr int kCentroidRadius = 2;

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
    cufftHandle planFwd = 0;
    cufftHandle planInv = 0;
    int srcWidth = 0;
    int srcHeight = 0;
    int dftWidth = 0;
    int dftHeight = 0;
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

    cudaError_t ensure(int width, int height, int paddedW, int paddedH) {
        const size_t nSrc = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t nDft = static_cast<size_t>(paddedW) * static_cast<size_t>(paddedH);
        const size_t specN = static_cast<size_t>(paddedH) * static_cast<size_t>(paddedW / 2 + 1);
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dTmplU8), &u8Cap, nSrc * 2));
        dDataU8 = dTmplU8 + nSrc;
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dTmpl), &realCap, nDft * sizeof(double) * 3));
        dData = dTmpl + nDft;
        dCorr = dTmpl + 2 * nDft;
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dFft1), &specCap,
                               specN * sizeof(cufftDoubleComplex) * 2));
        dFft2 = dFft1 + specN;

        if (dftWidth != paddedW || dftHeight != paddedH || planFwd == 0 || planInv == 0) {
            if (planFwd) {
                cufftDestroy(planFwd);
                planFwd = 0;
            }
            if (planInv) {
                cufftDestroy(planInv);
                planInv = 0;
            }
            CUFFT_CHECK(cufftPlan2d(&planFwd, paddedH, paddedW, CUFFT_D2Z));
            CUFFT_CHECK(cufftPlan2d(&planInv, paddedH, paddedW, CUFFT_Z2D));
            dftWidth = paddedW;
            dftHeight = paddedH;
        }
        srcWidth = width;
        srcHeight = height;
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
        dTmplU8 = dDataU8 = nullptr;
        dTmpl = dData = dCorr = nullptr;
        dFft1 = dFft2 = nullptr;
        u8Cap = realCap = specCap = 0;
        srcWidth = srcHeight = dftWidth = dftHeight = 0;
    }
};

DeviceWorkspace& defaultWorkspace() {
    static DeviceWorkspace workspace;
    return workspace;
}

DeviceWorkspace& workspaceFrom(void* ptr) {
    return ptr ? *static_cast<DeviceWorkspace*>(ptr) : defaultWorkspace();
}

// Same factorization as cv::getOptimalDFTSize: 2^p * 3^q * 5^r.
int optimalDftSize(int n0) {
    if (n0 <= 1) {
        return 1;
    }
    const int limit = n0 * 2 + 32;
    for (int n = n0; n < limit; ++n) {
        int x = n;
        while ((x % 2) == 0) {
            x /= 2;
        }
        while ((x % 3) == 0) {
            x /= 3;
        }
        while ((x % 5) == 0) {
            x /= 5;
        }
        if (x == 1) {
            return n;
        }
    }
    return n0;
}

__global__ void padAndWindowKernel(const unsigned char* __restrict__ src, double* __restrict__ dst,
                                   int srcW, int srcH, int dftW, int dftH, int useHanning) {
    const int n = dftW * dftH;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % dftW;
        const int y = i / dftW;
        const double val = (x < srcW && y < srcH) ? static_cast<double>(src[y * srcW + x]) : 0.0;
        if (!useHanning) {
            dst[i] = val;
            continue;
        }
        const double wx =
            dftW > 1 ? 0.5 * (1.0 - cos(2.0 * kPi * static_cast<double>(x) / (dftW - 1))) : 1.0;
        const double wy =
            dftH > 1 ? 0.5 * (1.0 - cos(2.0 * kPi * static_cast<double>(y) / (dftH - 1))) : 1.0;
        // OpenCV createHanningWindow stores Hann then sqrt()s it.
        dst[i] = val * sqrt(wx * wy);
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

int launchGrid(int n) {
    const int grid = (n + kBlockSize - 1) / kBlockSize;
    return grid < 1 ? 1 : grid;
}

void fftShift2D(std::vector<double>& data, int rows, int cols) {
    if (rows == 1 && cols == 1) {
        return;
    }

    const int xMid = cols >> 1;
    const int yMid = rows >> 1;
    const int isXodd = cols % 2;
    const int isYodd = rows % 2;
    std::vector<double> tmp = data;

    auto copyRect = [&](int srcX, int srcY, int dstX, int dstY, int w, int h) {
        for (int y = 0; y < h; ++y) {
            std::memcpy(&data[static_cast<size_t>(dstY + y) * cols + dstX],
                        &tmp[static_cast<size_t>(srcY + y) * cols + srcX],
                        static_cast<size_t>(w) * sizeof(double));
        }
    };

    if (!(isXodd || isYodd)) {
        copyRect(0, 0, xMid, yMid, xMid, yMid);
        copyRect(xMid, yMid, 0, 0, xMid, yMid);
        copyRect(xMid, 0, 0, yMid, xMid, yMid);
        copyRect(0, yMid, xMid, 0, xMid, yMid);
        return;
    }

    copyRect(0, 0, xMid, yMid, xMid + isXodd, yMid + isYodd);
    copyRect(xMid + isXodd, yMid + isYodd, 0, 0, xMid, yMid);
    copyRect(xMid + isXodd, 0, 0, yMid, xMid, yMid + isYodd);
    copyRect(0, yMid + isYodd, xMid, 0, xMid + isXodd, yMid);
}

void weightedCentroid(const std::vector<double>& src, int rows, int cols, int peakX, int peakY,
                      double* outX, double* outY, double* response) {
    const int minr = std::max(0, peakY - kCentroidRadius);
    const int maxr = std::min(rows - 1, peakY + kCentroidRadius);
    const int minc = std::max(0, peakX - kCentroidRadius);
    const int maxc = std::min(cols - 1, peakX + kCentroidRadius);

    double cx = 0.0;
    double cy = 0.0;
    double sum = 0.0;
    for (int y = minr; y <= maxr; ++y) {
        const double* row = src.data() + static_cast<size_t>(y) * cols;
        for (int x = minc; x <= maxc; ++x) {
            const double v = row[x];
            cx += static_cast<double>(x) * v;
            cy += static_cast<double>(y) * v;
            sum += v;
        }
    }
    if (response) {
        *response = sum;
    }
    sum += kEps;
    *outX = cx / sum;
    *outY = cy / sum;
}

cudaError_t computeDevice(DeviceWorkspace& ws, const unsigned char* templateImg,
                          const unsigned char* dataImg, int width, int height,
                          PhaseCorrelateResult* result, bool useHanning) {
    if (!templateImg || !dataImg || !result || width <= 1 || height <= 1) {
        return cudaErrorInvalidValue;
    }
    memset(result, 0, sizeof(*result));

    const int dftW = optimalDftSize(width);
    const int dftH = optimalDftSize(height);
    CUDA_CHECK(ws.ensure(width, height, dftW, dftH));

    const size_t nSrc = static_cast<size_t>(width) * static_cast<size_t>(height);
    const int nDft = dftW * dftH;
    const int specN = dftH * (dftW / 2 + 1);
    CUDA_CHECK(cudaMemcpy(ws.dTmplU8, templateImg, nSrc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.dDataU8, dataImg, nSrc, cudaMemcpyHostToDevice));

    padAndWindowKernel<<<launchGrid(nDft), kBlockSize>>>(ws.dTmplU8, ws.dTmpl, width, height, dftW,
                                                         dftH, useHanning ? 1 : 0);
    padAndWindowKernel<<<launchGrid(nDft), kBlockSize>>>(ws.dDataU8, ws.dData, width, height, dftW,
                                                         dftH, useHanning ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecD2Z(ws.planFwd, ws.dTmpl, ws.dFft1));
    CUFFT_CHECK(cufftExecD2Z(ws.planFwd, ws.dData, ws.dFft2));
    crossPowerKernel<<<launchGrid(specN), kBlockSize>>>(ws.dFft1, ws.dFft2, ws.dFft1, specN);
    CUDA_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftExecZ2D(ws.planInv, ws.dFft1, ws.dCorr));

    std::vector<double> corr(static_cast<size_t>(nDft));
    CUDA_CHECK(cudaMemcpy(corr.data(), ws.dCorr, corr.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    fftShift2D(corr, dftH, dftW);

    int peak = 0;
    double peakVal = corr[0];
    for (int i = 1; i < nDft; ++i) {
        if (corr[i] > peakVal) {
            peakVal = corr[i];
            peak = i;
        }
    }

    const int peakX = peak % dftW;
    const int peakY = peak / dftW;
    double subX = 0.0;
    double subY = 0.0;
    double response = 0.0;
    weightedCentroid(corr, dftH, dftW, peakX, peakY, &subX, &subY, &response);
    response /= static_cast<double>(dftW) * static_cast<double>(dftH);

    result->dx = static_cast<double>(dftW) / 2.0 - subX;
    result->dy = static_cast<double>(dftH) / 2.0 - subY;
    result->response = response;
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
    return computeDevice(defaultWorkspace(), templateImg, dataImg, width, height, result, true);
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
                                        PhaseCorrelateResult& result, bool useHanning) const {
    return computeDevice(workspaceFrom(deviceWorkspace_), templateImg, dataImg, width, height,
                         &result, useHanning);
}
