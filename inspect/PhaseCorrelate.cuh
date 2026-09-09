#pragma once

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PhaseCorrelateResult {
    double dx;
    double dy;
    double response;
} PhaseCorrelateResult;

cudaError_t phaseCorrelateCompute(const unsigned char* templateImg, const unsigned char* dataImg,
                                  int width, int height, PhaseCorrelateResult* result);

void phaseCorrelateReleaseWorkspace(void);

#ifdef __cplusplus
}

class PhaseCorrelateCuda {
public:
    PhaseCorrelateCuda();
    ~PhaseCorrelateCuda();

    PhaseCorrelateCuda(const PhaseCorrelateCuda&) = delete;
    PhaseCorrelateCuda& operator=(const PhaseCorrelateCuda&) = delete;

    cudaError_t compute(const unsigned char* templateImg, const unsigned char* dataImg, int width,
                        int height, PhaseCorrelateResult& result) const;

private:
    mutable void* deviceWorkspace_;
};

#endif
