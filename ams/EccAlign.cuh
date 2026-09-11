#pragma once

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EccCudaParams {
    int maxIters;
    double eps;
    int gaussFiltSize;
    int motionType;  // 0=MOTION_TRANSLATION, 1=MOTION_EUCLIDEAN
} EccCudaParams;

typedef struct EccCudaResult {
    double dx;
    double dy;
    double cc;
    int iterations;
    int ok;
} EccCudaResult;

EccCudaParams eccCudaDefaultParams(void);

cudaError_t eccAlignCompute(const unsigned char* templateImg, const unsigned char* dataImg,
                            int width, int height, const EccCudaParams* params,
                            EccCudaResult* result);

void eccAlignReleaseWorkspace(void);

#ifdef __cplusplus
}

class EccAlignCuda {
public:
    EccAlignCuda();
    explicit EccAlignCuda(const EccCudaParams& params);
    ~EccAlignCuda();

    EccAlignCuda(const EccAlignCuda&) = delete;
    EccAlignCuda& operator=(const EccAlignCuda&) = delete;

    const EccCudaParams& params() const { return params_; }
    void setParams(const EccCudaParams& params) { params_ = params; }

    cudaError_t compute(const unsigned char* templateImg, const unsigned char* dataImg, int width,
                        int height, EccCudaResult& result) const;

private:
    EccCudaParams params_;
    mutable void* deviceWorkspace_;
};

#endif
