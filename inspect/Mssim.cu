#include "Mssim.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace mssim_cuda_detail {

constexpr int kBlockSize = 256;
constexpr int kRadius = 5;
constexpr int kKernelSize = 11;
constexpr int kBorder = 5;
constexpr float kSigma = 1.5f;
constexpr float k1 = 0.01f;
constexpr float k2 = 0.03f;
constexpr float kL = 255.0f;
constexpr float kC1 = (k1 * kL) * (k1 * kL);
constexpr float kC2 = (k2 * kL) * (k2 * kL);

#define CUDA_CHECK(call)                 \
    do {                                 \
        const cudaError_t _err = (call); \
        if (_err != cudaSuccess) {       \
            return _err;                 \
        }                                \
    } while (0)

__constant__ float cKernel[kKernelSize];
static std::mutex gKernelMutex;
static std::vector<int> gKernelDevices;

cudaError_t ensureKernel() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(gKernelMutex);
    if (std::find(gKernelDevices.begin(), gKernelDevices.end(), device) !=
        gKernelDevices.end()) {
        return cudaSuccess;
    }
    float host[kKernelSize];
    float sum = 0.0f;
    const float twoSigma2 = 2.0f * kSigma * kSigma;
    for (int i = -kRadius; i <= kRadius; ++i) {
        const float v = std::exp(-(static_cast<float>(i * i)) / twoSigma2);
        host[i + kRadius] = v;
        sum += v;
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < kKernelSize; ++i) {
        host[i] *= inv;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(cKernel, host, sizeof(host)));
    gKernelDevices.push_back(device);
    return cudaSuccess;
}

struct DeviceWorkspace {
    unsigned char* dU8a = nullptr;
    unsigned char* dU8b = nullptr;
    unsigned char* dFused = nullptr;
    float* dF1 = nullptr;
    float* dF2 = nullptr;
    float* dP12 = nullptr;
    float* dTmp = nullptr;
    float* dMu1 = nullptr;
    float* dMu2 = nullptr;
    float* dMap = nullptr;
    float* dAcc = nullptr;
    double* dSum = nullptr;
    size_t u8Cap = 0;
    size_t fusedCap = 0;
    size_t floatCap = 0;
    size_t accCap = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned long long generation = 0;

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

    cudaError_t ensureScratch(int w, int h, int channelCount) {
        const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t u8Bytes = n * static_cast<size_t>(channelCount);
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dFused), &fusedCap, u8Bytes));
        // Seven scratch planes plus the final accumulated map. f1/f2 are overwritten
        // in-place with their squared-and-blurred products after mu1/mu2 are ready.
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dF1), &floatCap,
                               n * sizeof(float) * 7));
        dF2 = dF1 + n;
        dP12 = dF1 + 2 * n;
        dTmp = dF1 + 3 * n;
        dMu1 = dF1 + 4 * n;
        dMu2 = dF1 + 5 * n;
        dMap = dF1 + 6 * n;
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dAcc), &accCap, n * sizeof(float)));
        if (!dSum) {
            CUDA_CHECK(cudaMalloc(&dSum, sizeof(double)));
        }
        return cudaSuccess;
    }

    cudaError_t ensureHostStaging(int w, int h, int channelCount) {
        CUDA_CHECK(ensureScratch(w, h, channelCount));
        const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t u8Bytes = n * static_cast<size_t>(channelCount);
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dU8a), &u8Cap, u8Bytes * 2));
        dU8b = dU8a + u8Bytes;
        return cudaSuccess;
    }

    size_t reservedBytes() const {
        return u8Cap + fusedCap + floatCap + accCap + (dSum ? sizeof(double) : 0);
    }

    void release() {
        cudaFree(dU8a);
        cudaFree(dFused);
        cudaFree(dF1);
        cudaFree(dAcc);
        cudaFree(dSum);
        dU8a = dU8b = dFused = nullptr;
        dF1 = dF2 = dP12 = dTmp = dMu1 = dMu2 = dMap = dAcc = nullptr;
        dSum = nullptr;
        u8Cap = fusedCap = floatCap = accCap = 0;
        width = height = channels = 0;
        ++generation;
    }
};

DeviceWorkspace& defaultWorkspace() {
    static DeviceWorkspace workspace;
    return workspace;
}

DeviceWorkspace& workspaceFrom(void* ptr) {
    return ptr ? *static_cast<DeviceWorkspace*>(ptr) : defaultWorkspace();
}

int launchGrid(int n) {
    const int grid = (n + kBlockSize - 1) / kBlockSize;
    return std::max(1, std::min(grid, 65535));
}

int reductionGrid(int n) { return std::min(launchGrid(n), 256); }

bool validShape(int width, int height, int channels) {
    if (width < kKernelSize || height < kKernelSize ||
        (channels != 1 && channels != 3)) {
        return false;
    }
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    return pixels <= static_cast<size_t>(INT_MAX) &&
           pixels * static_cast<size_t>(channels) <= static_cast<size_t>(INT_MAX);
}

__device__ int reflect101(int p, int n) {
    if (n <= 1) {
        return 0;
    }
    if (p < 0) {
        p = -p;
    } else if (p >= n) {
        p = 2 * n - 2 - p;
    }
    if (p < 0) {
        p = 0;
    } else if (p >= n) {
        p = n - 1;
    }
    return p;
}

__global__ void u8PlaneToFloatKernel(const unsigned char* __restrict__ src, float* __restrict__ dst,
                                     int nPixels, int channels, int channel) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPixels; i += blockDim.x * gridDim.x) {
        dst[i] = static_cast<float>(src[i * channels + channel]);
    }
}

__global__ void productsInPlaceKernel(float* __restrict__ a, float* __restrict__ b,
                                      float* __restrict__ ab, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float av = a[i];
        const float bv = b[i];
        ab[i] = av * bv;
        a[i] = av * av;
        b[i] = bv * bv;
    }
}

__global__ void gaussHorizontalKernel(const float* __restrict__ src, float* __restrict__ dst, int w,
                                      int h) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        float acc = 0.0f;
#pragma unroll
        for (int k = -kRadius; k <= kRadius; ++k) {
            const int xx = reflect101(x + k, w);
            acc += src[y * w + xx] * cKernel[k + kRadius];
        }
        dst[i] = acc;
    }
}

__global__ void gaussVerticalKernel(const float* __restrict__ src, float* __restrict__ dst, int w,
                                    int h) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        float acc = 0.0f;
#pragma unroll
        for (int k = -kRadius; k <= kRadius; ++k) {
            const int yy = reflect101(y + k, h);
            acc += src[yy * w + x] * cKernel[k + kRadius];
        }
        dst[i] = acc;
    }
}

__global__ void ssimMapKernel(const float* __restrict__ mu1, const float* __restrict__ mu2,
                              const float* __restrict__ g11, const float* __restrict__ g22,
                              const float* __restrict__ g12, float* __restrict__ map, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float m1 = mu1[i];
        const float m2 = mu2[i];
        const float mu1Sq = m1 * m1;
        const float mu2Sq = m2 * m2;
        const float mu1Mu2 = m1 * m2;
        const float sigma1Sq = g11[i] - mu1Sq;
        const float sigma2Sq = g22[i] - mu2Sq;
        const float sigma12 = g12[i] - mu1Mu2;
        const float num = (2.0f * mu1Mu2 + kC1) * (2.0f * sigma12 + kC2);
        const float den = (mu1Sq + mu2Sq + kC1) * (sigma1Sq + sigma2Sq + kC2);
        map[i] = num / den;
    }
}

__global__ void addScaledKernel(float* __restrict__ acc, const float* __restrict__ src, int n,
                                float scale) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        acc[i] += src[i] * scale;
    }
}

__global__ void zeroKernel(float* dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = 0.0f;
    }
}

__global__ void sumInnerBlockKernel(const float* __restrict__ map, int w, int innerWidth,
                                    int innerPixels, int border, double* __restrict__ sum) {
    double local = 0.0;
    for (int inner = blockIdx.x * blockDim.x + threadIdx.x; inner < innerPixels;
         inner += blockDim.x * gridDim.x) {
        const int x = inner % innerWidth + border;
        const int y = inner / innerWidth + border;
        local += static_cast<double>(map[y * w + x]);
    }

    __shared__ double blockSum[kBlockSize];
    blockSum[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            blockSum[threadIdx.x] += blockSum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sum, blockSum[0]);
    }
}

__global__ void finalizeMeanKernel(double* mean, double invCount) {
    if (threadIdx.x == 0) {
        *mean *= invCount;
    }
}

__global__ void fuseKernel(const unsigned char* __restrict__ tmpl,
                           const unsigned char* __restrict__ aligned, const float* __restrict__ weight,
                           unsigned char* __restrict__ out, int nPixels, int channels) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPixels; i += blockDim.x * gridDim.x) {
        const float w = weight[i];
        const float iw = 1.0f - w;
        for (int c = 0; c < channels; ++c) {
            const int idx = i * channels + c;
            const float v = static_cast<float>(tmpl[idx]) * w + static_cast<float>(aligned[idx]) * iw;
            const int iv = static_cast<int>(v + (v >= 0.0f ? 0.5f : -0.5f));
            out[idx] = static_cast<unsigned char>(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
        }
    }
}

cudaError_t blur(DeviceWorkspace& ws, const float* src, float* dst, int n) {
    gaussHorizontalKernel<<<launchGrid(n), kBlockSize>>>(src, ws.dTmp, ws.width, ws.height);
    CUDA_CHECK(cudaGetLastError());
    gaussVerticalKernel<<<launchGrid(n), kBlockSize>>>(ws.dTmp, dst, ws.width, ws.height);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t computeChannel(DeviceWorkspace& ws, const unsigned char* srcA,
                           const unsigned char* srcB, int n, int channels, int channel,
                           float* mapOut) {
    u8PlaneToFloatKernel<<<launchGrid(n), kBlockSize>>>(srcA, ws.dF1, n, channels, channel);
    u8PlaneToFloatKernel<<<launchGrid(n), kBlockSize>>>(srcB, ws.dF2, n, channels, channel);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(blur(ws, ws.dF1, ws.dMu1, n));
    CUDA_CHECK(blur(ws, ws.dF2, ws.dMu2, n));

    productsInPlaceKernel<<<launchGrid(n), kBlockSize>>>(ws.dF1, ws.dF2, ws.dP12, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(blur(ws, ws.dF1, ws.dF1, n));
    CUDA_CHECK(blur(ws, ws.dF2, ws.dF2, n));
    CUDA_CHECK(blur(ws, ws.dP12, ws.dP12, n));

    ssimMapKernel<<<launchGrid(n), kBlockSize>>>(ws.dMu1, ws.dMu2, ws.dF1, ws.dF2, ws.dP12,
                                                 mapOut, n);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t reduceMeanDevice(DeviceWorkspace& ws, const float* map, int w, int h,
                             const double** meanDeviceOut) {
    CUDA_CHECK(cudaMemset(ws.dSum, 0, sizeof(double)));
    const int innerWidth = w - 2 * kBorder;
    const int innerHeight = h - 2 * kBorder;
    const int innerPixels = innerWidth * innerHeight;
    sumInnerBlockKernel<<<reductionGrid(innerPixels), kBlockSize>>>(
        map, w, innerWidth, innerPixels, kBorder, ws.dSum);
    CUDA_CHECK(cudaGetLastError());
    finalizeMeanKernel<<<1, 1>>>(ws.dSum, 1.0 / static_cast<double>(innerPixels));
    CUDA_CHECK(cudaGetLastError());
    *meanDeviceOut = ws.dSum;
    return cudaSuccess;
}

cudaError_t computeResident(DeviceWorkspace& ws, const unsigned char* image1Device,
                            const unsigned char* image2Device, int width, int height,
                            int channels, const float** mapDeviceOut,
                            const double** mssimDeviceOut) {
    if (!image1Device || !image2Device || !mapDeviceOut || !mssimDeviceOut ||
        !validShape(width, height, channels)) {
        return cudaErrorInvalidValue;
    }
    *mapDeviceOut = nullptr;
    *mssimDeviceOut = nullptr;
    ++ws.generation;
    ws.width = ws.height = ws.channels = 0;
    CUDA_CHECK(ensureKernel());
    CUDA_CHECK(ws.ensureScratch(width, height, channels));

    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (channels == 1) {
        CUDA_CHECK(computeChannel(ws, image1Device, image2Device, static_cast<int>(n), 1, 0,
                                  ws.dAcc));
    } else {
        zeroKernel<<<launchGrid(static_cast<int>(n)), kBlockSize>>>(ws.dAcc, static_cast<int>(n));
        CUDA_CHECK(cudaGetLastError());
        for (int c = 0; c < 3; ++c) {
            CUDA_CHECK(computeChannel(ws, image1Device, image2Device, static_cast<int>(n), 3, c,
                                      ws.dMap));
            addScaledKernel<<<launchGrid(static_cast<int>(n)), kBlockSize>>>(
                ws.dAcc, ws.dMap, static_cast<int>(n), 1.0f / 3.0f);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(reduceMeanDevice(ws, ws.dAcc, width, height, mssimDeviceOut));
    ws.width = width;
    ws.height = height;
    ws.channels = channels;
    *mapDeviceOut = ws.dAcc;
    return cudaSuccess;
}

cudaError_t fuseResident(DeviceWorkspace& ws, const unsigned char* templateDevice,
                         const unsigned char* alignedDevice, const float* mapDevice, int width,
                         int height, int channels, const unsigned char** fusedDeviceOut) {
    if (!templateDevice || !alignedDevice || !mapDevice || !fusedDeviceOut ||
        !validShape(width, height, channels)) {
        return cudaErrorInvalidValue;
    }

    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    fuseKernel<<<launchGrid(static_cast<int>(n)), kBlockSize>>>(
        templateDevice, alignedDevice, mapDevice, ws.dFused, static_cast<int>(n), channels);
    CUDA_CHECK(cudaGetLastError());
    *fusedDeviceOut = ws.dFused;
    return cudaSuccess;
}

cudaError_t downloadResidentView(DeviceWorkspace& ws, const MssimDeviceView& view,
                                 float* mapOut, unsigned char* fusedOut,
                                 double* mssimOut) {
    if (!view.ssimMap || !view.mssimDevice || !view.fusion || !mapOut || !fusedOut ||
        !mssimOut || !validShape(view.width, view.height, view.channels) ||
        view.owner != &ws || view.generation != ws.generation ||
        view.width != ws.width || view.height != ws.height ||
        view.channels != ws.channels || view.ssimMap != ws.dAcc ||
        view.mssimDevice != ws.dSum || view.fusion != ws.dFused) {
        return cudaErrorInvalidValue;
    }
    const size_t n = static_cast<size_t>(view.width) * static_cast<size_t>(view.height);
    const size_t u8Bytes = n * static_cast<size_t>(view.channels);
    CUDA_CHECK(cudaMemcpy(mapOut, view.ssimMap, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mssimOut, view.mssimDevice, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fusedOut, view.fusion, u8Bytes, cudaMemcpyDeviceToHost));
    return cudaSuccess;
}

cudaError_t computeHost(DeviceWorkspace& ws, const unsigned char* image1,
                        const unsigned char* image2, int width, int height, int channels,
                        float* mapOut, double* mssimOut) {
    if (!image1 || !image2 || !mapOut || !mssimOut ||
        !validShape(width, height, channels)) {
        return cudaErrorInvalidValue;
    }
    ++ws.generation;
    ws.width = ws.height = ws.channels = 0;
    CUDA_CHECK(ws.ensureHostStaging(width, height, channels));
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t u8Bytes = n * static_cast<size_t>(channels);
    CUDA_CHECK(cudaMemcpy(ws.dU8a, image1, u8Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.dU8b, image2, u8Bytes, cudaMemcpyHostToDevice));
    const float* mapDevice = nullptr;
    const double* mssimDevice = nullptr;
    CUDA_CHECK(computeResident(ws, ws.dU8a, ws.dU8b, width, height, channels, &mapDevice,
                               &mssimDevice));
    CUDA_CHECK(cudaMemcpy(mssimOut, mssimDevice, sizeof(double), cudaMemcpyDeviceToHost));
    return cudaMemcpy(mapOut, mapDevice, n * sizeof(float), cudaMemcpyDeviceToHost);
}

cudaError_t fuseHost(DeviceWorkspace& ws, const unsigned char* templateImg,
                     const unsigned char* alignedImg, const float* ssimMap, int width,
                     int height, int channels, unsigned char* fusedOut) {
    if (!templateImg || !alignedImg || !ssimMap || !fusedOut ||
        !validShape(width, height, channels)) {
        return cudaErrorInvalidValue;
    }
    ++ws.generation;
    ws.width = ws.height = ws.channels = 0;
    CUDA_CHECK(ws.ensureHostStaging(width, height, channels));
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t u8Bytes = n * static_cast<size_t>(channels);
    CUDA_CHECK(cudaMemcpy(ws.dU8a, templateImg, u8Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.dU8b, alignedImg, u8Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.dAcc, ssimMap, n * sizeof(float), cudaMemcpyHostToDevice));
    const unsigned char* fusedDevice = nullptr;
    CUDA_CHECK(fuseResident(ws, ws.dU8a, ws.dU8b, ws.dAcc, width, height, channels,
                            &fusedDevice));
    CUDA_CHECK(cudaMemcpy(fusedOut, fusedDevice, u8Bytes, cudaMemcpyDeviceToHost));
    return cudaSuccess;
}

#undef CUDA_CHECK

}  // namespace mssim_cuda_detail

using mssim_cuda_detail::DeviceWorkspace;
using mssim_cuda_detail::computeHost;
using mssim_cuda_detail::computeResident;
using mssim_cuda_detail::defaultWorkspace;
using mssim_cuda_detail::downloadResidentView;
using mssim_cuda_detail::fuseHost;
using mssim_cuda_detail::fuseResident;
using mssim_cuda_detail::workspaceFrom;

cudaError_t mssimCompute(const unsigned char* image1, const unsigned char* image2, int width,
                         int height, int channels, float* mapOut, MssimCudaResult* result) {
    if (!result) {
        return cudaErrorInvalidValue;
    }
    return computeHost(defaultWorkspace(), image1, image2, width, height, channels, mapOut,
                       &result->mssim);
}

cudaError_t mssimFuse(const unsigned char* templateImg, const unsigned char* alignedImg,
                      const float* ssimMap, int width, int height, int channels,
                      unsigned char* fusedOut) {
    return fuseHost(defaultWorkspace(), templateImg, alignedImg, ssimMap, width, height, channels,
                    fusedOut);
}

void mssimReleaseWorkspace(void) { defaultWorkspace().release(); }

MssimCuda::MssimCuda() : deviceWorkspace_(new DeviceWorkspace()) {}

MssimCuda::~MssimCuda() {
    if (deviceWorkspace_) {
        static_cast<DeviceWorkspace*>(deviceWorkspace_)->release();
        delete static_cast<DeviceWorkspace*>(deviceWorkspace_);
        deviceWorkspace_ = nullptr;
    }
}

cudaError_t MssimCuda::compute(const unsigned char* image1, const unsigned char* image2, int width,
                               int height, int channels, float* mapOut, double& mssim) const {
    return computeHost(workspaceFrom(deviceWorkspace_), image1, image2, width, height, channels,
                       mapOut, &mssim);
}

cudaError_t MssimCuda::fuse(const unsigned char* templateImg, const unsigned char* alignedImg,
                            const float* ssimMap, int width, int height, int channels,
                            unsigned char* fusedOut) const {
    return fuseHost(workspaceFrom(deviceWorkspace_), templateImg, alignedImg, ssimMap, width,
                    height, channels, fusedOut);
}

cudaError_t MssimCuda::computeResident(const unsigned char* image1Device,
                                       const unsigned char* image2Device, int width, int height,
                                       int channels, MssimDeviceView& view) const {
    view = {};
    auto& ws = workspaceFrom(deviceWorkspace_);
    const float* mapDevice = nullptr;
    const double* mssimDevice = nullptr;
    const cudaError_t status = mssim_cuda_detail::computeResident(
        ws, image1Device, image2Device, width, height, channels, &mapDevice, &mssimDevice);
    if (status != cudaSuccess) {
        return status;
    }
    view.ssimMap = mapDevice;
    view.mssimDevice = mssimDevice;
    view.fusion = nullptr;
    view.width = width;
    view.height = height;
    view.channels = channels;
    view.owner = &ws;
    view.generation = ws.generation;
    return cudaSuccess;
}

cudaError_t MssimCuda::fuseResident(const unsigned char* templateDevice,
                                    const unsigned char* alignedDevice,
                                    MssimDeviceView& view) const {
    view.fusion = nullptr;
    auto& ws = workspaceFrom(deviceWorkspace_);
    if (!templateDevice || !alignedDevice || !view.ssimMap || !view.mssimDevice ||
        !mssim_cuda_detail::validShape(view.width, view.height, view.channels) ||
        view.owner != &ws || view.generation != ws.generation ||
        view.width != ws.width || view.height != ws.height || view.channels != ws.channels ||
        view.ssimMap != ws.dAcc || view.mssimDevice != ws.dSum) {
        return cudaErrorInvalidValue;
    }
    const unsigned char* fusedDevice = nullptr;
    const cudaError_t status = mssim_cuda_detail::fuseResident(
        ws, templateDevice, alignedDevice, view.ssimMap, view.width, view.height,
        view.channels, &fusedDevice);
    if (status == cudaSuccess) {
        view.fusion = fusedDevice;
    }
    return status;
}

cudaError_t MssimCuda::downloadResident(const MssimDeviceView& view, float* mapOut,
                                        unsigned char* fusedOut, double& mssim) const {
    return downloadResidentView(workspaceFrom(deviceWorkspace_), view, mapOut, fusedOut,
                                &mssim);
}

size_t MssimCuda::workspaceBytes() const {
    return workspaceFrom(deviceWorkspace_).reservedBytes();
}
