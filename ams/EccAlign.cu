#include "EccAlign.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ecc_cuda_detail {

constexpr int kBlockSize = 256;
constexpr int kMaxRadius = 7;
constexpr int kInterpBits = 5;
constexpr int kInterpTab = 1 << kInterpBits;  // OpenCV INTER_TAB_SIZE=32

#define CUDA_CHECK(call)                 \
    do {                                 \
        const cudaError_t _err = (call); \
        if (_err != cudaSuccess) {       \
            return _err;                 \
        }                                \
    } while (0)

__constant__ float cGauss[2 * kMaxRadius + 1];

struct ReducePack {
    double count;
    double sumW;
    double sumT;
    double sumWW;
    double sumTT;
    double sumTW;
    double sumJ0;
    double sumJ1;
    double sumJ2;
    double sumJ0J0;
    double sumJ0J1;
    double sumJ0J2;
    double sumJ1J1;
    double sumJ1J2;
    double sumJ2J2;
    double sumJ0W;
    double sumJ1W;
    double sumJ2W;
    double sumJ0T;
    double sumJ1T;
    double sumJ2T;
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
        width = w;
        height = h;
        return cudaSuccess;
    }

    void release() {
        cudaFree(dTmplU8);
        cudaFree(dMask);
        cudaFree(dTmpl);
        cudaFree(dRed);
        dTmplU8 = dDataU8 = dMask = nullptr;
        dTmpl = dData = dTmp = dGx = dGy = dWarped = dGxW = dGyW = nullptr;
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
    float host[2 * kMaxRadius + 1] = {};
    // OpenCV getGaussianKernel: sigma<=0 and odd ksize<=7 uses a fixed kernel.
    static const float kSmallGauss[][7] = {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f},
    };
    if (ksize >= 1 && ksize <= 7 && (ksize % 2) == 1) {
        const float* src = kSmallGauss[ksize >> 1];
        for (int i = 0; i < ksize; ++i) {
            host[i] = src[i];
        }
    } else {
        const double sigma = 0.3 * ((static_cast<double>(ksize) - 1.0) * 0.5 - 1.0) + 0.8;
        const double twoSigma2 = 2.0 * sigma * sigma;
        double hostD[2 * kMaxRadius + 1] = {};
        double sum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double v = std::exp(-(static_cast<double>(i * i)) / twoSigma2);
            hostD[i + radius] = v;
            sum += v;
        }
        const double inv = 1.0 / sum;
        for (int i = 0; i < 2 * radius + 1; ++i) {
            host[i] = static_cast<float>(hostD[i] * inv);
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(cGauss, host, sizeof(host)));
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

__device__ float sampleOpenCvLinear(const float* src, int w, int h, float sx, float sy) {
    // OpenCV warpAffine INTER_LINEAR: saturate_cast<int>(s * INTER_TAB_SIZE) then 5-bit weights.
    const int X = cvRoundDevice(sx * static_cast<float>(kInterpTab));
    const int Y = cvRoundDevice(sy * static_cast<float>(kInterpTab));
    const int x0 = X >> kInterpBits;
    const int y0 = Y >> kInterpBits;
    const float fx = static_cast<float>(X & (kInterpTab - 1)) * (1.0f / static_cast<float>(kInterpTab));
    const float fy = static_cast<float>(Y & (kInterpTab - 1)) * (1.0f / static_cast<float>(kInterpTab));
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
                                      int h, int radius) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            acc += src[y * w + reflect101(x + k, w)] * cGauss[k + radius];
        }
        dst[i] = acc;
    }
}

__global__ void gaussVerticalKernel(const float* __restrict__ src, float* __restrict__ dst, int w,
                                    int h, int radius) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            acc += src[reflect101(y + k, h) * w + x] * cGauss[k + radius];
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

__global__ void warpAffineKernel(const float* __restrict__ src, const float* __restrict__ gx,
                                 const float* __restrict__ gy, float* __restrict__ dst,
                                 float* __restrict__ dgx, float* __restrict__ dgy,
                                 unsigned char* __restrict__ mask, int w, int h, float h00,
                                 float h01, float h02, float h10, float h11, float h12) {
    const int n = w * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        // OpenCV ECC: warpAffine(..., INTER_LINEAR + WARP_INVERSE_MAP)
        const float sx = h00 * static_cast<float>(x) + h01 * static_cast<float>(y) + h02;
        const float sy = h10 * static_cast<float>(x) + h11 * static_cast<float>(y) + h12;
        const int nx = cvRoundDevice(sx);
        const int ny = cvRoundDevice(sy);
        mask[i] = (nx >= 0 && nx < w && ny >= 0 && ny < h) ? 1 : 0;
        dst[i] = sampleOpenCvLinear(src, w, h, sx, sy);
        dgx[i] = sampleOpenCvLinear(gx, w, h, sx, sy);
        dgy[i] = sampleOpenCvLinear(gy, w, h, sx, sy);
    }
}

__device__ void addPack(ReducePack& a, const ReducePack& b) {
    a.count += b.count;
    a.sumW += b.sumW;
    a.sumT += b.sumT;
    a.sumWW += b.sumWW;
    a.sumTT += b.sumTT;
    a.sumTW += b.sumTW;
    a.sumJ0 += b.sumJ0;
    a.sumJ1 += b.sumJ1;
    a.sumJ2 += b.sumJ2;
    a.sumJ0J0 += b.sumJ0J0;
    a.sumJ0J1 += b.sumJ0J1;
    a.sumJ0J2 += b.sumJ0J2;
    a.sumJ1J1 += b.sumJ1J1;
    a.sumJ1J2 += b.sumJ1J2;
    a.sumJ2J2 += b.sumJ2J2;
    a.sumJ0W += b.sumJ0W;
    a.sumJ1W += b.sumJ1W;
    a.sumJ2W += b.sumJ2W;
    a.sumJ0T += b.sumJ0T;
    a.sumJ1T += b.sumJ1T;
    a.sumJ2T += b.sumJ2T;
}

__global__ void meanKernel(const float* __restrict__ warped, const float* __restrict__ tmpl,
                           const unsigned char* __restrict__ mask, int n, ReducePack* red) {
    ReducePack local{};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (!mask[i]) {
            continue;
        }
        const double wv = warped[i];
        const double t = tmpl[i];
        local.count += 1.0;
        local.sumW += wv;
        local.sumT += t;
        local.sumWW += wv * wv;
        local.sumTT += t * t;
        local.sumTW += t * wv;
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
    }
}

__global__ void projKernel(const float* __restrict__ warped, const float* __restrict__ tmpl,
                           const float* __restrict__ gx, const float* __restrict__ gy,
                           const unsigned char* __restrict__ mask, int w, int h, int euclidean,
                           float cosT, float sinT, float meanW, float meanT, ReducePack* red) {
    const int n = w * h;
    ReducePack local{};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int x = i % w;
        const int y = i / w;
        const bool m = mask[i] != 0;
        const float wv = warped[i];
        const float t = tmpl[i];
        // OpenCV: subtract mean only where mask!=0; templateZM is 0 outside the mask.
        const float wUse = m ? (wv - meanW) : wv;
        const float tUse = m ? (t - meanT) : 0.0f;
        const float j1 = gx[i];
        const float j2 = gy[i];
        float j0 = 0.0f;
        if (euclidean) {
            const float xf = static_cast<float>(x);
            const float yf = static_cast<float>(y);
            const float hatX = -sinT * xf - cosT * yf;
            const float hatY = cosT * xf - sinT * yf;
            j0 = j1 * hatX + j2 * hatY;
        }
        local.sumJ0J0 += static_cast<double>(j0) * static_cast<double>(j0);
        local.sumJ0J1 += static_cast<double>(j0) * static_cast<double>(j1);
        local.sumJ0J2 += static_cast<double>(j0) * static_cast<double>(j2);
        local.sumJ1J1 += static_cast<double>(j1) * static_cast<double>(j1);
        local.sumJ1J2 += static_cast<double>(j1) * static_cast<double>(j2);
        local.sumJ2J2 += static_cast<double>(j2) * static_cast<double>(j2);
        local.sumJ0W += static_cast<double>(j0) * static_cast<double>(wUse);
        local.sumJ1W += static_cast<double>(j1) * static_cast<double>(wUse);
        local.sumJ2W += static_cast<double>(j2) * static_cast<double>(wUse);
        local.sumJ0T += static_cast<double>(j0) * static_cast<double>(tUse);
        local.sumJ1T += static_cast<double>(j1) * static_cast<double>(tUse);
        local.sumJ2T += static_cast<double>(j2) * static_cast<double>(tUse);
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
        atomicAdd(&red->sumJ0J0, shared[0].sumJ0J0);
        atomicAdd(&red->sumJ0J1, shared[0].sumJ0J1);
        atomicAdd(&red->sumJ0J2, shared[0].sumJ0J2);
        atomicAdd(&red->sumJ1J1, shared[0].sumJ1J1);
        atomicAdd(&red->sumJ1J2, shared[0].sumJ1J2);
        atomicAdd(&red->sumJ2J2, shared[0].sumJ2J2);
        atomicAdd(&red->sumJ0W, shared[0].sumJ0W);
        atomicAdd(&red->sumJ1W, shared[0].sumJ1W);
        atomicAdd(&red->sumJ2W, shared[0].sumJ2W);
        atomicAdd(&red->sumJ0T, shared[0].sumJ0T);
        atomicAdd(&red->sumJ1T, shared[0].sumJ1T);
        atomicAdd(&red->sumJ2T, shared[0].sumJ2T);
    }
}

cudaError_t blur(DeviceWorkspace& ws, float* srcDst, int n, int ksize) {
    const int radius = ksize / 2;
    gaussHorizontalKernel<<<launchGrid(n), kBlockSize>>>(srcDst, ws.dTmp, ws.width, ws.height,
                                                         radius);
    CUDA_CHECK(cudaGetLastError());
    gaussVerticalKernel<<<launchGrid(n), kBlockSize>>>(ws.dTmp, srcDst, ws.width, ws.height,
                                                       radius);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

bool invert2x2(float h00, float h01, float h11, float* i00, float* i01, float* i11) {
    const double det = static_cast<double>(h00) * h11 - static_cast<double>(h01) * h01;
    if (std::abs(det) < 1e-12) {
        return false;
    }
    const double invDet = 1.0 / det;
    *i00 = static_cast<float>(h11 * invDet);
    *i11 = static_cast<float>(h00 * invDet);
    *i01 = static_cast<float>(-h01 * invDet);
    return true;
}

bool invert3x3(const float H[9], float inv[9]) {
    const double a00 = H[0];
    const double a01 = H[1];
    const double a02 = H[2];
    const double a10 = H[3];
    const double a11 = H[4];
    const double a12 = H[5];
    const double a20 = H[6];
    const double a21 = H[7];
    const double a22 = H[8];
    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a12 * a20 - a10 * a22;
    const double c02 = a10 * a21 - a11 * a20;
    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (std::abs(det) < 1e-12) {
        return false;
    }
    const double invDet = 1.0 / det;
    inv[0] = static_cast<float>(c00 * invDet);
    inv[1] = static_cast<float>((a02 * a21 - a01 * a22) * invDet);
    inv[2] = static_cast<float>((a01 * a12 - a02 * a11) * invDet);
    inv[3] = static_cast<float>(c01 * invDet);
    inv[4] = static_cast<float>((a00 * a22 - a02 * a20) * invDet);
    inv[5] = static_cast<float>((a02 * a10 - a00 * a12) * invDet);
    inv[6] = static_cast<float>(c02 * invDet);
    inv[7] = static_cast<float>((a01 * a20 - a00 * a21) * invDet);
    inv[8] = static_cast<float>((a00 * a11 - a01 * a10) * invDet);
    return true;
}

void centerImageMoments(ReducePack* s) {
    const double n = s->count;
    const double meanW = s->sumW / n;
    const double meanT = s->sumT / n;
    s->sumWW -= n * meanW * meanW;
    s->sumTT -= n * meanT * meanT;
    s->sumTW -= n * meanW * meanT;
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
    const bool euclidean = params.motionType == 1;  // cv::MOTION_EUCLIDEAN
    if (params.motionType != 0 && params.motionType != 1) {
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

    // OpenCV 2x3 identity: [cos -sin tx; sin cos ty]
    float h00 = 1.0f;
    float h01 = 0.0f;
    float h02 = 0.0f;
    float h10 = 0.0f;
    float h11 = 1.0f;
    float h12 = 0.0f;
    double rho = -1.0;
    double lastRho = -params.eps;
    int iter = 0;

    for (iter = 1; iter <= params.maxIters && std::abs(rho - lastRho) >= params.eps; ++iter) {
        warpAffineKernel<<<launchGrid(n), kBlockSize>>>(ws.dData, ws.dGx, ws.dGy, ws.dWarped,
                                                        ws.dGxW, ws.dGyW, ws.dMask, width, height,
                                                        h00, h01, h02, h10, h11, h12);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(ws.dRed, 0, sizeof(ReducePack)));
        meanKernel<<<launchGrid(n), kBlockSize>>>(ws.dWarped, ws.dTmpl, ws.dMask, n, ws.dRed);
        CUDA_CHECK(cudaGetLastError());

        ReducePack mean{};
        CUDA_CHECK(cudaMemcpy(&mean, ws.dRed, sizeof(ReducePack), cudaMemcpyDeviceToHost));
        if (mean.count < 16.0) {
            result->ok = 0;
            return cudaSuccess;
        }
        const float meanW = static_cast<float>(mean.sumW / mean.count);
        const float meanT = static_cast<float>(mean.sumT / mean.count);
        centerImageMoments(&mean);

        const double imgNorm = std::sqrt(std::max(mean.sumWW, 0.0));
        const double tmpNorm = std::sqrt(std::max(mean.sumTT, 0.0));
        const double denom = imgNorm * tmpNorm;
        lastRho = rho;
        rho = denom > 1e-15 ? mean.sumTW / denom : -1.0;
        if (!std::isfinite(rho)) {
            result->ok = 0;
            return cudaSuccess;
        }

        CUDA_CHECK(cudaMemset(ws.dRed, 0, sizeof(ReducePack)));
        projKernel<<<launchGrid(n), kBlockSize>>>(ws.dWarped, ws.dTmpl, ws.dGxW, ws.dGyW, ws.dMask,
                                                  width, height, euclidean ? 1 : 0, h00, h10, meanW,
                                                  meanT, ws.dRed);
        CUDA_CHECK(cudaGetLastError());
        ReducePack s{};
        CUDA_CHECK(cudaMemcpy(&s, ws.dRed, sizeof(ReducePack), cudaMemcpyDeviceToHost));
        s.sumWW = mean.sumWW;
        s.sumTW = mean.sumTW;

        if (!euclidean) {
            // OpenCV stores Hessian / projections as CV_32F then Mat::inv().
            float i00 = 0.0f;
            float i01 = 0.0f;
            float i11 = 0.0f;
            if (!invert2x2(static_cast<float>(s.sumJ1J1), static_cast<float>(s.sumJ1J2),
                           static_cast<float>(s.sumJ2J2), &i00, &i01, &i11)) {
                result->ok = 0;
                return cudaSuccess;
            }
            const float j1w = static_cast<float>(s.sumJ1W);
            const float j2w = static_cast<float>(s.sumJ2W);
            const float j1t = static_cast<float>(s.sumJ1T);
            const float j2t = static_cast<float>(s.sumJ2T);
            const float ip0 = i00 * j1w + i01 * j2w;
            const float ip1 = i01 * j1w + i11 * j2w;
            const float lambdaN = static_cast<float>(s.sumWW) - (j1w * ip0 + j2w * ip1);
            const float lambdaD = static_cast<float>(s.sumTW) - (j1t * ip0 + j2t * ip1);
            if (lambdaD <= 0.0f) {
                result->ok = 0;
                result->cc = -1.0;
                return cudaSuccess;
            }
            const float lambda = lambdaN / lambdaD;
            const float ep0 = lambda * j1t - j1w;
            const float ep1 = lambda * j2t - j2w;
            h02 += i00 * ep0 + i01 * ep1;
            h12 += i01 * ep0 + i11 * ep1;
        } else {
            const float H[9] = {static_cast<float>(s.sumJ0J0), static_cast<float>(s.sumJ0J1),
                                static_cast<float>(s.sumJ0J2), static_cast<float>(s.sumJ0J1),
                                static_cast<float>(s.sumJ1J1), static_cast<float>(s.sumJ1J2),
                                static_cast<float>(s.sumJ0J2), static_cast<float>(s.sumJ1J2),
                                static_cast<float>(s.sumJ2J2)};
            float Hinv[9] = {};
            if (!invert3x3(H, Hinv)) {
                result->ok = 0;
                return cudaSuccess;
            }
            const float j0w = static_cast<float>(s.sumJ0W);
            const float j1w = static_cast<float>(s.sumJ1W);
            const float j2w = static_cast<float>(s.sumJ2W);
            const float j0t = static_cast<float>(s.sumJ0T);
            const float j1t = static_cast<float>(s.sumJ1T);
            const float j2t = static_cast<float>(s.sumJ2T);
            const float ip0 = Hinv[0] * j0w + Hinv[1] * j1w + Hinv[2] * j2w;
            const float ip1 = Hinv[3] * j0w + Hinv[4] * j1w + Hinv[5] * j2w;
            const float ip2 = Hinv[6] * j0w + Hinv[7] * j1w + Hinv[8] * j2w;
            const float lambdaN = static_cast<float>(s.sumWW) - (j0w * ip0 + j1w * ip1 + j2w * ip2);
            const float lambdaD = static_cast<float>(s.sumTW) - (j0t * ip0 + j1t * ip1 + j2t * ip2);
            if (lambdaD <= 0.0f) {
                result->ok = 0;
                result->cc = -1.0;
                return cudaSuccess;
            }
            const float lambda = lambdaN / lambdaD;
            const float ep0 = lambda * j0t - j0w;
            const float ep1 = lambda * j1t - j1w;
            const float ep2 = lambda * j2t - j2w;
            const float d0 = Hinv[0] * ep0 + Hinv[1] * ep1 + Hinv[2] * ep2;
            const float d1 = Hinv[3] * ep0 + Hinv[4] * ep1 + Hinv[5] * ep2;
            const float d2 = Hinv[6] * ep0 + Hinv[7] * ep1 + Hinv[8] * ep2;
            // OpenCV update_warping_matrix_ECC for MOTION_EUCLIDEAN
            const float sinArg = std::max(-1.0f, std::min(1.0f, h10));
            const double newTheta = static_cast<double>(d0) + std::asin(static_cast<double>(sinArg));
            h02 += d1;
            h12 += d2;
            const float c = static_cast<float>(std::cos(newTheta));
            const float sn = static_cast<float>(std::sin(newTheta));
            h00 = c;
            h11 = c;
            h10 = sn;
            h01 = -sn;
        }
    }

    result->dx = static_cast<double>(h02);
    result->dy = static_cast<double>(h12);
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
    params.motionType = 1;  // cv::MOTION_EUCLIDEAN
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
