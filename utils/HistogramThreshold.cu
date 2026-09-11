#include "HistogramThreshold.cuh"

#include <cstring>

// CUDA path: rgbToGray / histogram / apply on device, then the same host peak/T rules as CPU.

namespace hist_thresh_detail {

constexpr int kBlockSize = 256;
constexpr int kMaxGridSize = 2048;
constexpr int kHistCopies = 8;

#define CUDA_CHECK(call)                  \
    do {                                  \
        const cudaError_t _err = (call);  \
        if (_err != cudaSuccess) {        \
            return _err;                  \
        }                                 \
    } while (0)

struct DeviceWorkspace {
    unsigned char* dBgr = nullptr;
    unsigned char* dGray = nullptr;
    unsigned char* dBin = nullptr;
    unsigned int* dHist = nullptr;
    size_t bgrCap = 0;
    size_t grayCap = 0;
    size_t binCap = 0;

    cudaError_t ensure(unsigned char** ptr, size_t* cap, size_t bytes) {
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

    cudaError_t ensureBgr(size_t bytes) { return ensure(&dBgr, &bgrCap, bytes); }
    cudaError_t ensureGray(size_t bytes) { return ensure(&dGray, &grayCap, bytes); }
    cudaError_t ensureBin(size_t bytes) { return ensure(&dBin, &binCap, bytes); }

    cudaError_t ensureHist() {
        if (dHist) {
            return cudaSuccess;
        }
        return cudaMalloc(&dHist, HISTOGRAM_THRESHOLD_BINS * sizeof(unsigned int));
    }

    void release() {
        cudaFree(dBgr);
        cudaFree(dGray);
        cudaFree(dBin);
        cudaFree(dHist);
        dBgr = dGray = dBin = nullptr;
        dHist = nullptr;
        bgrCap = grayCap = binCap = 0;
    }
};

DeviceWorkspace& defaultWorkspace() {
    static DeviceWorkspace workspace;
    return workspace;
}

DeviceWorkspace& workspaceFrom(void* ptr) {
    return ptr ? *static_cast<DeviceWorkspace*>(ptr) : defaultWorkspace();
}

int launchGrid(int nPixels, int pixelsPerThread) {
    const int grid = (nPixels + kBlockSize * pixelsPerThread - 1) / (kBlockSize * pixelsPerThread);
    if (grid < 1) {
        return 1;
    }
    return grid > kMaxGridSize ? kMaxGridSize : grid;
}

__global__ void rgbToGrayKernel(const unsigned char* __restrict__ bgr,
                                unsigned char* __restrict__ gray, int nPixels) {
    const uchar3* src = reinterpret_cast<const uchar3*>(bgr);
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPixels; i += stride) {
        const uchar3 p = src[i];
        gray[i] = static_cast<unsigned char>((77 * p.z + 150 * p.y + 29 * p.x) >> 8);
    }
}

__global__ void histogramKernel(const unsigned char* __restrict__ gray, int nPixels,
                                unsigned int* __restrict__ hist) {
    __shared__ unsigned int localHist[kHistCopies * HISTOGRAM_THRESHOLD_BINS];
    for (int i = threadIdx.x; i < kHistCopies * HISTOGRAM_THRESHOLD_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    const int copy = threadIdx.x & (kHistCopies - 1);
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPixels; i += stride) {
        atomicAdd(&localHist[copy * HISTOGRAM_THRESHOLD_BINS + gray[i]], 1u);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < HISTOGRAM_THRESHOLD_BINS; bin += blockDim.x) {
        unsigned int sum = 0;
#pragma unroll
        for (int c = 0; c < kHistCopies; ++c) {
            sum += localHist[c * HISTOGRAM_THRESHOLD_BINS + bin];
        }
        if (sum != 0) {
            atomicAdd(&hist[bin], sum);
        }
    }
}

__global__ void bgrToGrayHistogramKernel(const unsigned char* __restrict__ bgr,
                                         unsigned char* __restrict__ gray, int nPixels,
                                         unsigned int* __restrict__ hist) {
    __shared__ unsigned int localHist[kHistCopies * HISTOGRAM_THRESHOLD_BINS];
    for (int i = threadIdx.x; i < kHistCopies * HISTOGRAM_THRESHOLD_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    const uchar3* src = reinterpret_cast<const uchar3*>(bgr);
    const int copy = threadIdx.x & (kHistCopies - 1);
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPixels; i += stride) {
        const uchar3 p = src[i];
        const unsigned char y = static_cast<unsigned char>((77 * p.z + 150 * p.y + 29 * p.x) >> 8);
        gray[i] = y;
        atomicAdd(&localHist[copy * HISTOGRAM_THRESHOLD_BINS + y], 1u);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < HISTOGRAM_THRESHOLD_BINS; bin += blockDim.x) {
        unsigned int sum = 0;
#pragma unroll
        for (int c = 0; c < kHistCopies; ++c) {
            sum += localHist[c * HISTOGRAM_THRESHOLD_BINS + bin];
        }
        if (sum != 0) {
            atomicAdd(&hist[bin], sum);
        }
    }
}

__global__ void applyKernel(const unsigned char* __restrict__ gray, unsigned char* __restrict__ binary,
                            int nPixels, int threshold) {
    const int stride = blockDim.x * gridDim.x;
    const int n4 = nPixels >> 2;
    const uchar4* src4 = reinterpret_cast<const uchar4*>(gray);
    uchar4* dst4 = reinterpret_cast<uchar4*>(binary);
    const unsigned char hi = 255;
    const unsigned char lo = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const uchar4 g = src4[i];
        uchar4 o;
        o.x = g.x > threshold ? hi : lo;
        o.y = g.y > threshold ? hi : lo;
        o.z = g.z > threshold ? hi : lo;
        o.w = g.w > threshold ? hi : lo;
        dst4[i] = o;
    }

    const int tail = n4 << 2;
    for (int i = tail + blockIdx.x * blockDim.x + threadIdx.x; i < nPixels; i += stride) {
        binary[i] = gray[i] > threshold ? hi : lo;
    }
}

cudaError_t rgbToGrayDevice(DeviceWorkspace& ws, const unsigned char* bgr, int nPixels,
                            unsigned char* grayOut) {
    const size_t bgrBytes = static_cast<size_t>(nPixels) * 3;
    CUDA_CHECK(ws.ensureBgr(bgrBytes));
    CUDA_CHECK(ws.ensureGray(nPixels));
    CUDA_CHECK(cudaMemcpy(ws.dBgr, bgr, bgrBytes, cudaMemcpyHostToDevice));

    rgbToGrayKernel<<<launchGrid(nPixels, 1), kBlockSize>>>(ws.dBgr, ws.dGray, nPixels);
    CUDA_CHECK(cudaGetLastError());
    return cudaMemcpy(grayOut, ws.dGray, nPixels, cudaMemcpyDeviceToHost);
}

cudaError_t histogramDevice(DeviceWorkspace& ws, const unsigned char* gray, int nPixels,
                            unsigned int* hostHist) {
    CUDA_CHECK(ws.ensureGray(nPixels));
    CUDA_CHECK(ws.ensureHist());
    CUDA_CHECK(cudaMemcpy(ws.dGray, gray, nPixels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ws.dHist, 0, HISTOGRAM_THRESHOLD_BINS * sizeof(unsigned int)));

    histogramKernel<<<launchGrid(nPixels, 16), kBlockSize>>>(ws.dGray, nPixels, ws.dHist);
    CUDA_CHECK(cudaGetLastError());
    return cudaMemcpy(hostHist, ws.dHist, HISTOGRAM_THRESHOLD_BINS * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost);
}

cudaError_t computeFromBgrDevice(DeviceWorkspace& ws, const unsigned char* bgr, int nPixels,
                                 unsigned char* grayOut, unsigned int* hostHist) {
    const size_t bgrBytes = static_cast<size_t>(nPixels) * 3;
    CUDA_CHECK(ws.ensureBgr(bgrBytes));
    CUDA_CHECK(ws.ensureGray(nPixels));
    CUDA_CHECK(ws.ensureHist());
    CUDA_CHECK(cudaMemcpy(ws.dBgr, bgr, bgrBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ws.dHist, 0, HISTOGRAM_THRESHOLD_BINS * sizeof(unsigned int)));

    bgrToGrayHistogramKernel<<<launchGrid(nPixels, 4), kBlockSize>>>(ws.dBgr, ws.dGray, nPixels,
                                                                     ws.dHist);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hostHist, ws.dHist, HISTOGRAM_THRESHOLD_BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    if (grayOut) {
        CUDA_CHECK(cudaMemcpy(grayOut, ws.dGray, nPixels, cudaMemcpyDeviceToHost));
    }
    return cudaSuccess;
}

cudaError_t applyDevice(DeviceWorkspace& ws, const unsigned char* gray, int nPixels, int threshold,
                        unsigned char* binaryOut) {
    CUDA_CHECK(ws.ensureGray(nPixels));
    CUDA_CHECK(ws.ensureBin(nPixels));
    CUDA_CHECK(cudaMemcpy(ws.dGray, gray, nPixels, cudaMemcpyHostToDevice));

    applyKernel<<<launchGrid(nPixels, 4), kBlockSize>>>(ws.dGray, ws.dBin, nPixels, threshold);
    CUDA_CHECK(cudaGetLastError());
    return cudaMemcpy(binaryOut, ws.dBin, nPixels, cudaMemcpyDeviceToHost);
}

cudaError_t computeFromBgrImpl(DeviceWorkspace& ws, const unsigned char* bgr, int width, int height,
                               const HistogramThresholdParams& params,
                               HistogramThresholdResult* result, unsigned char* grayOut) {
    if (!bgr || !result || width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }
    memset(result, 0, sizeof(*result));

    const int nPixels = width * height;
    unsigned int hist[HISTOGRAM_THRESHOLD_BINS];
    CUDA_CHECK(computeFromBgrDevice(ws, bgr, nPixels, grayOut, hist));
    histogramThresholdComputeFromHistogram(hist, nPixels, params, result);
    return cudaSuccess;
}

cudaError_t computeFromGrayImpl(DeviceWorkspace& ws, const unsigned char* gray, int width, int height,
                                const HistogramThresholdParams& params,
                                HistogramThresholdResult* result) {
    if (!gray || !result || width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }
    memset(result, 0, sizeof(*result));

    const int nPixels = width * height;
    unsigned int hist[HISTOGRAM_THRESHOLD_BINS];
    CUDA_CHECK(histogramDevice(ws, gray, nPixels, hist));
    histogramThresholdComputeFromHistogram(hist, nPixels, params, result);
    return cudaSuccess;
}

#undef CUDA_CHECK

}  // namespace hist_thresh_detail

using hist_thresh_detail::DeviceWorkspace;
using hist_thresh_detail::applyDevice;
using hist_thresh_detail::computeFromBgrImpl;
using hist_thresh_detail::computeFromGrayImpl;
using hist_thresh_detail::defaultWorkspace;
using hist_thresh_detail::histogramDevice;
using hist_thresh_detail::rgbToGrayDevice;
using hist_thresh_detail::workspaceFrom;

void histogramThresholdReleaseWorkspace(void) { defaultWorkspace().release(); }

cudaError_t histogramThresholdRgbToGray(const unsigned char* bgr, int nPixels,
                                        unsigned char* gray) {
    if (!bgr || !gray || nPixels <= 0) {
        return cudaErrorInvalidValue;
    }
    return rgbToGrayDevice(defaultWorkspace(), bgr, nPixels, gray);
}

cudaError_t histogramThresholdComputeHistogram(const unsigned char* gray, int nPixels,
                                               unsigned int histogram[HISTOGRAM_THRESHOLD_BINS]) {
    if (!gray || !histogram || nPixels <= 0) {
        return cudaErrorInvalidValue;
    }
    return histogramDevice(defaultWorkspace(), gray, nPixels, histogram);
}

cudaError_t histogramThresholdComputeFromBgr(const unsigned char* bgr, int width, int height,
                                             const HistogramThresholdParams* params,
                                             HistogramThresholdResult* result,
                                             unsigned char* grayOut) {
    const HistogramThresholdParams localParams =
        params ? *params : histogramThresholdDefaultParams();
    return computeFromBgrImpl(defaultWorkspace(), bgr, width, height, localParams, result, grayOut);
}

cudaError_t histogramThresholdComputeFromGray(const unsigned char* gray, int width, int height,
                                              const HistogramThresholdParams* params,
                                              HistogramThresholdResult* result) {
    const HistogramThresholdParams localParams =
        params ? *params : histogramThresholdDefaultParams();
    return computeFromGrayImpl(defaultWorkspace(), gray, width, height, localParams, result);
}

cudaError_t histogramThresholdApply(const unsigned char* gray, int nPixels, int threshold,
                                    unsigned char* binaryOut) {
    if (!gray || !binaryOut || nPixels <= 0) {
        return cudaErrorInvalidValue;
    }
    return applyDevice(defaultWorkspace(), gray, nPixels, threshold, binaryOut);
}

HistogramThreshold::HistogramThreshold()
    : params_(histogramThresholdDefaultParams()), deviceWorkspace_(new DeviceWorkspace()) {}

HistogramThreshold::HistogramThreshold(const HistogramThresholdParams& params)
    : params_(params), deviceWorkspace_(new DeviceWorkspace()) {}

HistogramThreshold::~HistogramThreshold() {
    if (deviceWorkspace_) {
        static_cast<DeviceWorkspace*>(deviceWorkspace_)->release();
        delete static_cast<DeviceWorkspace*>(deviceWorkspace_);
        deviceWorkspace_ = nullptr;
    }
}

cudaError_t HistogramThreshold::rgbToGray(const unsigned char* bgr, int nPixels,
                                          unsigned char* gray) const {
    return rgbToGrayDevice(workspaceFrom(deviceWorkspace_), bgr, nPixels, gray);
}

cudaError_t HistogramThreshold::computeHistogram(
    const unsigned char* gray, int nPixels, unsigned int histogram[HISTOGRAM_THRESHOLD_BINS]) const {
    return histogramDevice(workspaceFrom(deviceWorkspace_), gray, nPixels, histogram);
}

cudaError_t HistogramThreshold::computeFromBgr(const unsigned char* bgr, int width, int height,
                                               HistogramThresholdResult& result,
                                               unsigned char* grayOut) const {
    return computeFromBgrImpl(workspaceFrom(deviceWorkspace_), bgr, width, height, params_, &result,
                              grayOut);
}

cudaError_t HistogramThreshold::computeFromGray(const unsigned char* gray, int width, int height,
                                                HistogramThresholdResult& result) const {
    return computeFromGrayImpl(workspaceFrom(deviceWorkspace_), gray, width, height, params_,
                               &result);
}

cudaError_t HistogramThreshold::apply(const unsigned char* gray, int nPixels, int threshold,
                                      unsigned char* binaryOut) const {
    return applyDevice(workspaceFrom(deviceWorkspace_), gray, nPixels, threshold, binaryOut);
}

bool HistogramThreshold::rgbToGrayCpu(const unsigned char* bgr, int nPixels,
                                      unsigned char* gray) const {
    return histogramThresholdRgbToGrayCpu(bgr, nPixels, gray) == 0;
}

bool HistogramThreshold::computeFromGrayCpu(const unsigned char* gray, int width, int height,
                                            HistogramThresholdResult& result) const {
    return histogramThresholdComputeFromGrayCpu(gray, width, height, &params_, &result) == 0;
}

bool HistogramThreshold::computeFromBgrCpu(const unsigned char* bgr, int width, int height,
                                           HistogramThresholdResult& result,
                                           unsigned char* grayOut) const {
    return histogramThresholdComputeFromBgrCpu(bgr, width, height, &params_, &result, grayOut) == 0;
}

bool HistogramThreshold::applyCpu(const unsigned char* gray, int nPixels, int threshold,
                                  unsigned char* binaryOut) const {
    return histogramThresholdApplyCpu(gray, nPixels, threshold, binaryOut) == 0;
}

int HistogramThreshold::fuse(int templateT, int dataT) {
    return histogramThresholdFuse(templateT, dataT);
}
