#include "EccAlign.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ecc_cuda_detail {

constexpr int kBlockSize = 256;
constexpr int kMaxRadius = 7;

#define CUDA_CHECK(call)                 \
    do {                                 \
        const cudaError_t _err = (call); \
        if (_err != cudaSuccess) {       \
            return _err;                 \
        }                                \
    } while (0)

struct ReducePack {
    double count;
    double sumW;
    double sumT;
    double sumWW;
    double sumTT;
    double sumTW;
    double sumGx;
    double sumGy;
    double sumGxGx;
    double sumGyGy;
    double sumGxGy;
    double sumGxW;
    double sumGyW;
    double sumGxT;
    double sumGyT;
};

struct DeviceWorkspace {
    unsigned char* dTmplU8 = nullptr;
    unsigned char* dDataU8 = nullptr;
    unsigned char* dMask = nullptr;
    float* dTmpl = nullptr;
    float* dData = nullptr;
    float* dTmp = nullptr;
    float* dGx = nullptr;
    float* dGy = nullptr;
    float* dWarped = nullptr;
    float* dGxW = nullptr;
    float* dGyW = nullptr;
    float* dGauss = nullptr;
    ReducePack* dRed = nullptr;
    size_t u8Cap = 0;
    size_t maskCap = 0;
    size_t floatCap = 0;
    int width = 0;
    int height = 0;
    int gaussSize = 0;

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
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dTmplU8), &u8Cap, n * 2));
        dDataU8 = dTmplU8 + n;
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dMask), &maskCap, n));
        CUDA_CHECK(ensureBytes(reinterpret_cast<void**>(&dTmpl), &floatCap, n * sizeof(float) * 8));
        dData = dTmpl + n;
        dTmp = dTmpl + 2 * n;
        dGx = dTmpl + 3 * n;
        dGy = dTmpl + 4 * n;
        dWarped = dTmpl + 5 * n;
        dGxW = dTmpl + 6 * n;
        dGyW = dTmpl + 7 * n;
        if (!dRed) {
            CUDA_CHECK(cudaMalloc(&dRed, sizeof(ReducePack)));
        }
        if (!dGauss) {
            CUDA_CHECK(cudaMalloc(&dGauss, sizeof(float) * (2 * kMaxRadius + 1)));
        }
        width = w;
        height = h;
        return cudaSuccess;
    }

    void release() {
        cudaFree(dTmplU8);
        cudaFree(dMask);
        cudaFree(dTmpl);
        cudaFree(dGauss);
        cudaFree(dRed);
        dTmplU8 = dDataU8 = dMask = nullptr;
        dTmpl = dData = dTmp = dGx = dGy = dWarped = dGxW = dGyW = nullptr;
        dGauss = nullptr;
        dRed = nullptr;
        u8Cap = maskCap = floatCap = 0;
        width = height = 0;
        gaussSize = 0;
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
    return grid < 1 ? 1 : grid;
}

cudaError_t uploadGauss(int ksize, DeviceWorkspace& ws) {
    if (ksize == ws.gaussSize) {
        return cudaSuccess;
    }
    const int radius = ksize / 2;
    const float sigma = 0.3f * ((static_cast<float>(ksize) - 1.0f) * 0.5f - 1.0f) + 0.8f;
    const float twoSigma2 = 2.0f * sigma * sigma;
    float host[2 * kMaxRadius + 1] = {};
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float v = std::exp(-(static_cast<float>(i * i)) / twoSigma2);
        host[i + radius] = v;
        sum += v;
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < 2 * radius + 1; ++i) {
        host[i] *= inv;
    }
    CUDA_CHECK(cudaMemcpy(ws.dGauss, host, sizeof(float) * static_cast<size_t>(ksize),
                          cudaMemcpyHostToDevice));
    ws.gaussSize = ksize;
    return cudaSuccess;
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

__device__ int cvRoundDevice(float value) {
    return static_cast<int>(value + (value >= 0.0f ? 0.5f : -0.5f));
}

__device__ float sampleAtOrZero(const float* src, int w, int h, int x, int y) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return 0.0f;
    }
    return src[y * w + x];
}

__device__ float sampleBilinearConst0(const float* src, int w, int h, float sx, float sy) {
    const int x0 = static_cast<int>(floorf(sx));
    const int y0 = static_cast<int>(floorf(sy));
    const float fx = sx - static_cast<float>(x0);
    const float fy = sy - static_cast<float>(y0);
    const float v00 = sampleAtOrZero(src, w, h, x0, y0);
    const float v01 = sampleAtOrZero(src, w, h, x0 + 1, y0);
    const float v10 = sampleAtOrZero(src, w, h, x0, y0 + 1);
    const float v11 = sampleAtOrZero(src, w, h, x0 + 1, y0 + 1);
    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) + fy * ((1.0f - fx) * v10 + fx * v11);
}

__global__ void u8ToFloatKernel(const unsigned char* __restrict__ src, float* __restrict__ dst,
                                int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = static_cast<float>(src[i]);
    }
}

__global__ void gaussHorizontalKernel(const float* __restrict__ src, float* __restrict__ dst, int w,
                                      int h, int radius, const float* __restrict__ gauss) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            acc += src[y * w + reflect101(x + k, w)] * gauss[k + radius];
        }
        dst[i] = acc;
    }
}

__global__ void gaussVerticalKernel(const float* __restrict__ src, float* __restrict__ dst, int w,
                                    int h, int radius, const float* __restrict__ gauss) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            acc += src[reflect101(y + k, h) * w + x] * gauss[k + radius];
        }
        dst[i] = acc;
    }
}

__global__ void gradientKernel(const float* __restrict__ src, float* __restrict__ gx,
                               float* __restrict__ gy, int w, int h) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        const float left = src[y * w + reflect101(x - 1, w)];
        const float right = src[y * w + reflect101(x + 1, w)];
        const float up = src[reflect101(y - 1, h) * w + x];
        const float down = src[reflect101(y + 1, h) * w + x];
        gx[i] = 0.5f * (right - left);
        gy[i] = 0.5f * (down - up);
    }
}

__global__ void warpTranslationKernel(const float* __restrict__ src, const float* __restrict__ gx,
                                      const float* __restrict__ gy, float* __restrict__ dst,
                                      float* __restrict__ dgx, float* __restrict__ dgy,
                                      unsigned char* __restrict__ mask, int w, int h, float tx,
                                      float ty) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        // OpenCV ECC: warpAffine(..., INTER_LINEAR + WARP_INVERSE_MAP)
        // dst(x, y) = src(x + tx, y + ty)
        const float sx = static_cast<float>(x) + tx;
        const float sy = static_cast<float>(y) + ty;
        const int nx = cvRoundDevice(sx);
        const int ny = cvRoundDevice(sy);
        const unsigned char inside = (nx >= 0 && nx < w && ny >= 0 && ny < h) ? 1 : 0;
        mask[i] = inside;
        dst[i] = sampleBilinearConst0(src, w, h, sx, sy);
        dgx[i] = sampleBilinearConst0(gx, w, h, sx, sy);
        dgy[i] = sampleBilinearConst0(gy, w, h, sx, sy);
    }
}

__device__ void addPack(ReducePack& a, const ReducePack& b) {
    a.count += b.count;
    a.sumW += b.sumW;
    a.sumT += b.sumT;
    a.sumWW += b.sumWW;
    a.sumTT += b.sumTT;
    a.sumTW += b.sumTW;
    a.sumGx += b.sumGx;
    a.sumGy += b.sumGy;
    a.sumGxGx += b.sumGxGx;
    a.sumGyGy += b.sumGyGy;
    a.sumGxGy += b.sumGxGy;
    a.sumGxW += b.sumGxW;
    a.sumGyW += b.sumGyW;
    a.sumGxT += b.sumGxT;
    a.sumGyT += b.sumGyT;
}

__global__ void statsKernel(const float* __restrict__ warped, const float* __restrict__ tmpl,
                            const float* __restrict__ gx, const float* __restrict__ gy,
                            const unsigned char* __restrict__ mask, int n, ReducePack* red) {
    ReducePack local{};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (!mask[i]) {
            continue;
        }
        const double w = warped[i];
        const double t = tmpl[i];
        const double ix = gx[i];
        const double iy = gy[i];
        local.count += 1.0;
        local.sumW += w;
        local.sumT += t;
        local.sumWW += w * w;
        local.sumTT += t * t;
        local.sumTW += t * w;
        local.sumGx += ix;
        local.sumGy += iy;
        local.sumGxGx += ix * ix;
        local.sumGyGy += iy * iy;
        local.sumGxGy += ix * iy;
        local.sumGxW += ix * w;
        local.sumGyW += iy * w;
        local.sumGxT += ix * t;
        local.sumGyT += iy * t;
    }

    __shared__ ReducePack shared[kBlockSize];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            addPack(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&red->count, shared[0].count);
        atomicAdd(&red->sumW, shared[0].sumW);
        atomicAdd(&red->sumT, shared[0].sumT);
        atomicAdd(&red->sumWW, shared[0].sumWW);
        atomicAdd(&red->sumTT, shared[0].sumTT);
        atomicAdd(&red->sumTW, shared[0].sumTW);
        atomicAdd(&red->sumGx, shared[0].sumGx);
        atomicAdd(&red->sumGy, shared[0].sumGy);
        atomicAdd(&red->sumGxGx, shared[0].sumGxGx);
        atomicAdd(&red->sumGyGy, shared[0].sumGyGy);
        atomicAdd(&red->sumGxGy, shared[0].sumGxGy);
        atomicAdd(&red->sumGxW, shared[0].sumGxW);
        atomicAdd(&red->sumGyW, shared[0].sumGyW);
        atomicAdd(&red->sumGxT, shared[0].sumGxT);
        atomicAdd(&red->sumGyT, shared[0].sumGyT);
    }
}

cudaError_t blur(DeviceWorkspace& ws, float* srcDst, int n, int ksize) {
    const int radius = ksize / 2;
    gaussHorizontalKernel<<<launchGrid(n), kBlockSize>>>(srcDst, ws.dTmp, ws.width, ws.height,
                                                         radius, ws.dGauss);
    CUDA_CHECK(cudaGetLastError());
    gaussVerticalKernel<<<launchGrid(n), kBlockSize>>>(ws.dTmp, srcDst, ws.width, ws.height,
                                                       radius, ws.dGauss);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

bool invert2x2(double h00, double h01, double h11, double* i00, double* i01, double* i11) {
    const double det = h00 * h11 - h01 * h01;
    if (std::abs(det) < 1e-12) {
        return false;
    }
    const double invDet = 1.0 / det;
    *i00 = h11 * invDet;
    *i11 = h00 * invDet;
    *i01 = -h01 * invDet;
    return true;
}

void centerMoments(ReducePack* s) {
    const double n = s->count;
    const double meanW = s->sumW / n;
    const double meanT = s->sumT / n;
    s->sumWW -= n * meanW * meanW;
    s->sumTT -= n * meanT * meanT;
    s->sumTW -= n * meanW * meanT;
    s->sumGxW -= meanW * s->sumGx;
    s->sumGyW -= meanW * s->sumGy;
    s->sumGxT -= meanT * s->sumGx;
    s->sumGyT -= meanT * s->sumGy;
}

cudaError_t computeDevice(DeviceWorkspace& ws, const unsigned char* templateImg,
                          const unsigned char* dataImg, int width, int height,
                          const EccCudaParams& params, EccCudaResult* result) {
    if (!templateImg || !dataImg || !result || width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }
    if (params.maxIters <= 0 || params.gaussFiltSize < 1 || (params.gaussFiltSize % 2) == 0 ||
        params.gaussFiltSize > 2 * kMaxRadius + 1) {
        return cudaErrorInvalidValue;
    }

    memset(result, 0, sizeof(*result));
    CUDA_CHECK(ws.ensure(width, height));
    CUDA_CHECK(uploadGauss(params.gaussFiltSize, ws));

    const int n = width * height;
    CUDA_CHECK(cudaMemcpy(ws.dTmplU8, templateImg, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.dDataU8, dataImg, n, cudaMemcpyHostToDevice));
    u8ToFloatKernel<<<launchGrid(n), kBlockSize>>>(ws.dTmplU8, ws.dTmpl, n);
    u8ToFloatKernel<<<launchGrid(n), kBlockSize>>>(ws.dDataU8, ws.dData, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(blur(ws, ws.dTmpl, n, params.gaussFiltSize));
    CUDA_CHECK(blur(ws, ws.dData, n, params.gaussFiltSize));
    gradientKernel<<<launchGrid(n), kBlockSize>>>(ws.dData, ws.dGx, ws.dGy, width, height);
    CUDA_CHECK(cudaGetLastError());

    float tx = 0.0f;
    float ty = 0.0f;
    double rho = -1.0;
    double lastRho = -params.eps;
    int iter = 0;

    for (iter = 1; iter <= params.maxIters && std::abs(rho - lastRho) >= params.eps; ++iter) {
        warpTranslationKernel<<<launchGrid(n), kBlockSize>>>(ws.dData, ws.dGx, ws.dGy, ws.dWarped,
                                                             ws.dGxW, ws.dGyW, ws.dMask, width,
                                                             height, tx, ty);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(ws.dRed, 0, sizeof(ReducePack)));
        statsKernel<<<launchGrid(n), kBlockSize>>>(ws.dWarped, ws.dTmpl, ws.dGxW, ws.dGyW, ws.dMask,
                                                   n, ws.dRed);
        CUDA_CHECK(cudaGetLastError());

        ReducePack s{};
        CUDA_CHECK(cudaMemcpy(&s, ws.dRed, sizeof(ReducePack), cudaMemcpyDeviceToHost));
        if (s.count < 16.0) {
            result->ok = 0;
            return cudaSuccess;
        }
        centerMoments(&s);

        const double imgNorm = std::sqrt(std::max(s.sumWW, 0.0));
        const double tmpNorm = std::sqrt(std::max(s.sumTT, 0.0));
        const double denom = imgNorm * tmpNorm;
        lastRho = rho;
        rho = denom > 1e-15 ? s.sumTW / denom : -1.0;
        if (!std::isfinite(rho)) {
            result->ok = 0;
            return cudaSuccess;
        }

        double i00 = 0.0;
        double i01 = 0.0;
        double i11 = 0.0;
        if (!invert2x2(s.sumGxGx, s.sumGxGy, s.sumGyGy, &i00, &i01, &i11)) {
            result->ok = 0;
            return cudaSuccess;
        }

        const double ip0 = i00 * s.sumGxW + i01 * s.sumGyW;
        const double ip1 = i01 * s.sumGxW + i11 * s.sumGyW;
        const double lambdaN = s.sumWW - (s.sumGxW * ip0 + s.sumGyW * ip1);
        const double lambdaD = s.sumTW - (s.sumGxT * ip0 + s.sumGyT * ip1);
        if (lambdaD <= 0.0) {
            result->ok = 0;
            result->cc = -1.0;
            return cudaSuccess;
        }
        const double lambda = lambdaN / lambdaD;
        const double ep0 = lambda * s.sumGxT - s.sumGxW;
        const double ep1 = lambda * s.sumGyT - s.sumGyW;
        tx = static_cast<float>(static_cast<double>(tx) + i00 * ep0 + i01 * ep1);
        ty = static_cast<float>(static_cast<double>(ty) + i01 * ep0 + i11 * ep1);
    }

    result->dx = static_cast<double>(tx);
    result->dy = static_cast<double>(ty);
    result->cc = rho;
    result->iterations = iter - 1;
    result->ok = 1;
    return cudaSuccess;
}

#undef CUDA_CHECK

}  // namespace ecc_cuda_detail

using ecc_cuda_detail::DeviceWorkspace;
using ecc_cuda_detail::computeDevice;
using ecc_cuda_detail::defaultWorkspace;
using ecc_cuda_detail::workspaceFrom;

EccCudaParams eccCudaDefaultParams(void) {
    EccCudaParams params;
    params.maxIters = 100;
    params.eps = 1e-5;
    params.gaussFiltSize = 5;
    return params;
}

cudaError_t eccAlignCompute(const unsigned char* templateImg, const unsigned char* dataImg,
                            int width, int height, const EccCudaParams* params,
                            EccCudaResult* result) {
    const EccCudaParams local = params ? *params : eccCudaDefaultParams();
    return computeDevice(defaultWorkspace(), templateImg, dataImg, width, height, local, result);
}

void eccAlignReleaseWorkspace(void) { defaultWorkspace().release(); }

EccAlignCuda::EccAlignCuda()
    : params_(eccCudaDefaultParams()), deviceWorkspace_(new DeviceWorkspace()) {}

EccAlignCuda::EccAlignCuda(const EccCudaParams& params)
    : params_(params), deviceWorkspace_(new DeviceWorkspace()) {}

EccAlignCuda::~EccAlignCuda() {
    if (deviceWorkspace_) {
        static_cast<DeviceWorkspace*>(deviceWorkspace_)->release();
        delete static_cast<DeviceWorkspace*>(deviceWorkspace_);
        deviceWorkspace_ = nullptr;
    }
}

cudaError_t EccAlignCuda::compute(const unsigned char* templateImg, const unsigned char* dataImg,
                                  int width, int height, EccCudaResult& result) const {
    return computeDevice(workspaceFrom(deviceWorkspace_), templateImg, dataImg, width, height,
                         params_, &result);
}
