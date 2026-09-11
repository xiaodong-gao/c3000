#include "IntensityNormalize.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace intensity_normalize_cuda_detail {

constexpr int kBlockSize = 256;

#define CUDA_CHECK(call)                 \
    do {                                 \
        const cudaError_t _err = (call); \
        if (_err != cudaSuccess) {       \
            return _err;                 \
        }                                \
    } while (0)

struct DeviceWorkspace {
    unsigned char* dBytes = nullptr;
    float* dL2 = nullptr;
    unsigned int* dHist = nullptr;
    double* dStats = nullptr;
    size_t byteCapacity = 0;
    size_t l2Capacity = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned long long generation = 0;

    cudaError_t ensure(int width, int height, int channels) {
        const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t imageBytes = pixels * static_cast<size_t>(channels);
        // template, data/normalized-data (in-place), and absolute difference. The
        // normalized template is an exact host-side alias and needs no extra buffer.
        const size_t requiredBytes = imageBytes * 3;
        if (byteCapacity < requiredBytes) {
            cudaFree(dBytes);
            dBytes = nullptr;
            byteCapacity = 0;
            CUDA_CHECK(cudaMalloc(&dBytes, requiredBytes));
            byteCapacity = requiredBytes;
        }
        if (l2Capacity < pixels) {
            cudaFree(dL2);
            dL2 = nullptr;
            l2Capacity = 0;
            CUDA_CHECK(cudaMalloc(&dL2, pixels * sizeof(float)));
            l2Capacity = pixels;
        }
        if (!dHist) {
            // template + data, up to 3 channels, 256 bins each.
            CUDA_CHECK(cudaMalloc(&dHist, sizeof(unsigned int) * 256 * 6));
        }
        if (!dStats) {
            CUDA_CHECK(cudaMalloc(&dStats, sizeof(double) * 24));
        }
        return cudaSuccess;
    }

    size_t reservedBytes() const {
        return byteCapacity + l2Capacity * sizeof(float) +
               (dHist ? sizeof(unsigned int) * 256 * 6 : 0) +
               (dStats ? sizeof(double) * 24 : 0);
    }

    void release() {
        cudaFree(dBytes);
        cudaFree(dL2);
        cudaFree(dHist);
        cudaFree(dStats);
        dBytes = nullptr;
        dL2 = nullptr;
        dHist = nullptr;
        dStats = nullptr;
        byteCapacity = 0;
        l2Capacity = 0;
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

int launchGrid(size_t count) {
    const size_t grid = (count + kBlockSize - 1) / kBlockSize;
    return static_cast<int>(std::max<size_t>(1, std::min<size_t>(grid, 65535)));
}

__global__ void interleavedHistogramKernel(const unsigned char* __restrict__ img, size_t pixels,
                                           int channels, unsigned int* __restrict__ hist) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < pixels;
         p += stride) {
        const size_t base = p * static_cast<size_t>(channels);
        for (int c = 0; c < channels; ++c) {
            atomicAdd(&hist[c * 256 + img[base + c]], 1u);
        }
    }
}

template <int Channels>
__global__ void normalizeAndDifferenceKernel(
    const unsigned char* __restrict__ templateRoi,
    const unsigned char* __restrict__ dataRoi, size_t pixels,
    const double* __restrict__ gain, const double* __restrict__ bias,
    unsigned char* __restrict__ dataNormalized,
    unsigned char* __restrict__ absoluteDifference, float* __restrict__ l2Difference) {
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < pixels;
         p += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t base = p * Channels;
        float sumSquared = 0.0f;
#pragma unroll
        for (int c = 0; c < Channels; ++c) {
            const unsigned char t = templateRoi[base + c];
            // OpenCV x64 saturate_cast uses SSE cvRound (round-to-nearest-even).
            const double mapped = static_cast<double>(dataRoi[base + c]) * gain[c] + bias[c];
            int iv = __double2int_rn(mapped);
            if (iv < 0) {
                iv = 0;
            } else if (iv > 255) {
                iv = 255;
            }
            const unsigned char d = static_cast<unsigned char>(iv);
            const int delta = static_cast<int>(t) - static_cast<int>(d);
            dataNormalized[base + c] = d;
            absoluteDifference[base + c] =
                static_cast<unsigned char>(delta < 0 ? -delta : delta);
            sumSquared += static_cast<float>(delta * delta);
        }
        l2Difference[p] = sqrtf(sumSquared);
    }
}

cudaError_t computeResidentWithWorkspace(
    DeviceWorkspace& workspace, const unsigned char* templateRoi,
    const unsigned char* dataRoi, int width, int height, int channels,
    IntensityNormalizeDeviceView* view) {
    if (!templateRoi || !dataRoi || !view || width <= 0 || height <= 0 ||
        (channels != 1 && channels != 3)) {
        return cudaErrorInvalidValue;
    }

    *view = {};
    ++workspace.generation;
    workspace.width = workspace.height = workspace.channels = 0;
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t imageBytes = pixels * static_cast<size_t>(channels);
    CUDA_CHECK(workspace.ensure(width, height, channels));
    unsigned char* dTemplate = workspace.dBytes;
    unsigned char* dData = dTemplate + imageBytes;
    // Normalization is element-wise, so dData can safely be overwritten in-place.
    unsigned char* dDataNormalized = dData;
    unsigned char* dAbsolute = dData + imageBytes;

    CUDA_CHECK(cudaMemcpy(dTemplate, templateRoi, imageBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dData, dataRoi, imageBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace.dHist, 0, sizeof(unsigned int) * 256 * 6));
    CUDA_CHECK(cudaMemset(workspace.dStats, 0, sizeof(double) * 24));

    unsigned int* dTemplateHist = workspace.dHist;
    unsigned int* dDataHist = workspace.dHist + channels * 256;
    interleavedHistogramKernel<<<launchGrid(pixels), kBlockSize>>>(dTemplate, pixels, channels,
                                                                   dTemplateHist);
    interleavedHistogramKernel<<<launchGrid(pixels), kBlockSize>>>(dData, pixels, channels,
                                                                   dDataHist);
    CUDA_CHECK(cudaGetLastError());

    unsigned int hostHist[256 * 6]{};
    CUDA_CHECK(cudaMemcpy(hostHist, workspace.dHist, sizeof(unsigned int) * 256 * channels * 2,
                          cudaMemcpyDeviceToHost));

    double hostStats[24]{};
    constexpr double kFlatEpsilon = 1e-9;
    for (int c = 0; c < channels; ++c) {
        double templateMean = 0.0;
        double templateStdDev = 0.0;
        double dataMean = 0.0;
        double dataStdDev = 0.0;
        robustMeanStdFromU8Histogram(hostHist + c * 256, static_cast<int>(pixels), templateMean,
                                     templateStdDev);
        robustMeanStdFromU8Histogram(hostHist + (channels + c) * 256, static_cast<int>(pixels),
                                     dataMean, dataStdDev);
        const double gain = dataStdDev > kFlatEpsilon ? templateStdDev / dataStdDev : 1.0;
        const double bias = templateMean - gain * dataMean;
        hostStats[c] = templateMean;
        hostStats[4 + c] = templateStdDev;
        hostStats[8 + c] = dataMean;
        hostStats[12 + c] = dataStdDev;
        hostStats[16 + c] = gain;
        hostStats[20 + c] = bias;
    }
    CUDA_CHECK(cudaMemcpy(workspace.dStats, hostStats, sizeof(hostStats), cudaMemcpyHostToDevice));

    if (channels == 1) {
        normalizeAndDifferenceKernel<1><<<launchGrid(pixels), kBlockSize>>>(
            dTemplate, dData, pixels, workspace.dStats + 16, workspace.dStats + 20,
            dDataNormalized, dAbsolute, workspace.dL2);
    } else {
        normalizeAndDifferenceKernel<3><<<launchGrid(pixels), kBlockSize>>>(
            dTemplate, dData, pixels, workspace.dStats + 16, workspace.dStats + 20,
            dDataNormalized, dAbsolute, workspace.dL2);
    }
    CUDA_CHECK(cudaGetLastError());

    workspace.width = width;
    workspace.height = height;
    workspace.channels = channels;
    view->templateRoi = dTemplate;
    view->dataNormalized = dDataNormalized;
    view->absoluteDifference = dAbsolute;
    view->l2Difference = workspace.dL2;
    view->statistics = workspace.dStats;
    view->width = width;
    view->height = height;
    view->channels = channels;
    view->owner = &workspace;
    view->generation = workspace.generation;
    return cudaSuccess;
}

cudaError_t downloadResidentView(DeviceWorkspace& workspace,
                                 const IntensityNormalizeDeviceView& view,
                                 unsigned char* dataNormalized,
                                 unsigned char* absoluteDifference, float* l2Difference,
                                 IntensityNormalizeCudaStats* stats) {
    if (!view.templateRoi || !view.dataNormalized || !view.absoluteDifference ||
        !view.l2Difference || !view.statistics || !dataNormalized || !absoluteDifference ||
        !l2Difference || !stats || view.width <= 0 || view.height <= 0 ||
        (view.channels != 1 && view.channels != 3) || view.owner != &workspace ||
        view.generation != workspace.generation || view.width != workspace.width ||
        view.height != workspace.height || view.channels != workspace.channels) {
        return cudaErrorInvalidValue;
    }
    const size_t pixels =
        static_cast<size_t>(view.width) * static_cast<size_t>(view.height);
    const size_t imageBytes = pixels * static_cast<size_t>(view.channels);
    const unsigned char* expectedTemplate = workspace.dBytes;
    const unsigned char* expectedNormalized = expectedTemplate + imageBytes;
    const unsigned char* expectedAbsolute = expectedNormalized + imageBytes;
    if (view.templateRoi != expectedTemplate || view.dataNormalized != expectedNormalized ||
        view.absoluteDifference != expectedAbsolute || view.l2Difference != workspace.dL2 ||
        view.statistics != workspace.dStats) {
        return cudaErrorInvalidValue;
    }
    CUDA_CHECK(cudaMemcpy(dataNormalized, view.dataNormalized, imageBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(absoluteDifference, view.absoluteDifference, imageBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(l2Difference, view.l2Difference, pixels * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double deviceStats[24]{};
    CUDA_CHECK(cudaMemcpy(deviceStats, view.statistics, sizeof(deviceStats),
                          cudaMemcpyDeviceToHost));
    std::memset(stats, 0, sizeof(*stats));
    stats->channels = view.channels;
    for (int c = 0; c < view.channels; ++c) {
        stats->templateMean[c] = deviceStats[c];
        stats->templateStdDev[c] = deviceStats[4 + c];
        stats->dataMean[c] = deviceStats[8 + c];
        stats->dataStdDev[c] = deviceStats[12 + c];
        stats->gain[c] = deviceStats[16 + c];
        stats->bias[c] = deviceStats[20 + c];
    }
    return cudaSuccess;
}

cudaError_t computeWithWorkspace(
    DeviceWorkspace& workspace, const unsigned char* templateRoi,
    const unsigned char* dataRoi, int width, int height, int channels,
    unsigned char* templateNormalized, unsigned char* dataNormalized,
    unsigned char* absoluteDifference, float* l2Difference,
    IntensityNormalizeCudaStats* stats) {
    if (!templateNormalized || !dataNormalized || !absoluteDifference || !l2Difference) {
        return cudaErrorInvalidValue;
    }
    IntensityNormalizeDeviceView view{};
    CUDA_CHECK(computeResidentWithWorkspace(workspace, templateRoi, dataRoi, width, height,
                                             channels, &view));
    const size_t imageBytes = static_cast<size_t>(width) * static_cast<size_t>(height) *
                              static_cast<size_t>(channels);
    // The reference is not transformed. Copying it directly avoids a redundant D2H transfer.
    std::memcpy(templateNormalized, templateRoi, imageBytes);
    return downloadResidentView(workspace, view, dataNormalized, absoluteDifference,
                                l2Difference, stats);
}

}  // namespace intensity_normalize_cuda_detail

extern "C" cudaError_t intensityNormalizeCompute(
    const unsigned char* templateRoi, const unsigned char* dataRoi, int width, int height,
    int channels, unsigned char* templateNormalized, unsigned char* dataNormalized,
    unsigned char* absoluteDifference, float* l2Difference,
    IntensityNormalizeCudaStats* stats) {
    return intensity_normalize_cuda_detail::computeWithWorkspace(
        intensity_normalize_cuda_detail::defaultWorkspace(), templateRoi, dataRoi, width,
        height, channels, templateNormalized, dataNormalized, absoluteDifference, l2Difference,
        stats);
}

extern "C" void intensityNormalizeReleaseWorkspace(void) {
    intensity_normalize_cuda_detail::defaultWorkspace().release();
}

IntensityNormalizeCuda::IntensityNormalizeCuda()
    : deviceWorkspace_(new intensity_normalize_cuda_detail::DeviceWorkspace()) {}

IntensityNormalizeCuda::~IntensityNormalizeCuda() {
    auto* workspace =
        static_cast<intensity_normalize_cuda_detail::DeviceWorkspace*>(deviceWorkspace_);
    if (workspace) {
        workspace->release();
        delete workspace;
    }
}

cudaError_t IntensityNormalizeCuda::compute(
    const unsigned char* templateRoi, const unsigned char* dataRoi, int width, int height,
    int channels, unsigned char* templateNormalized, unsigned char* dataNormalized,
    unsigned char* absoluteDifference, float* l2Difference,
    IntensityNormalizeStats& stats) const {
    IntensityNormalizeCudaStats cudaStats{};
    auto& workspace = intensity_normalize_cuda_detail::workspaceFrom(deviceWorkspace_);
    const cudaError_t status = intensity_normalize_cuda_detail::computeWithWorkspace(
        workspace, templateRoi, dataRoi, width, height, channels, templateNormalized,
        dataNormalized, absoluteDifference, l2Difference, &cudaStats);
    if (status != cudaSuccess) {
        return status;
    }
    stats.channels = cudaStats.channels;
    for (int c = 0; c < 4; ++c) {
        stats.templateMean[c] = cudaStats.templateMean[c];
        stats.templateStdDev[c] = cudaStats.templateStdDev[c];
        stats.dataMean[c] = cudaStats.dataMean[c];
        stats.dataStdDev[c] = cudaStats.dataStdDev[c];
        stats.gain[c] = cudaStats.gain[c];
        stats.bias[c] = cudaStats.bias[c];
    }
    return cudaSuccess;
}

cudaError_t IntensityNormalizeCuda::computeResident(
    const unsigned char* templateRoi, const unsigned char* dataRoi, int width, int height,
    int channels, IntensityNormalizeDeviceView& view) const {
    auto& workspace = intensity_normalize_cuda_detail::workspaceFrom(deviceWorkspace_);
    const cudaError_t status = intensity_normalize_cuda_detail::computeResidentWithWorkspace(
        workspace, templateRoi, dataRoi, width, height, channels, &view);
    if (status != cudaSuccess) {
        view = {};
    }
    return status;
}

cudaError_t IntensityNormalizeCuda::downloadResident(
    const IntensityNormalizeDeviceView& view, unsigned char* dataNormalized,
    unsigned char* absoluteDifference, float* l2Difference,
    IntensityNormalizeStats& stats) const {
    IntensityNormalizeCudaStats cudaStats{};
    const cudaError_t status = intensity_normalize_cuda_detail::downloadResidentView(
        intensity_normalize_cuda_detail::workspaceFrom(deviceWorkspace_), view,
        dataNormalized, absoluteDifference, l2Difference, &cudaStats);
    if (status != cudaSuccess) {
        return status;
    }
    stats.channels = cudaStats.channels;
    for (int c = 0; c < 4; ++c) {
        stats.templateMean[c] = cudaStats.templateMean[c];
        stats.templateStdDev[c] = cudaStats.templateStdDev[c];
        stats.dataMean[c] = cudaStats.dataMean[c];
        stats.dataStdDev[c] = cudaStats.dataStdDev[c];
        stats.gain[c] = cudaStats.gain[c];
        stats.bias[c] = cudaStats.bias[c];
    }
    return cudaSuccess;
}

size_t IntensityNormalizeCuda::workspaceBytes() const {
    return intensity_normalize_cuda_detail::workspaceFrom(deviceWorkspace_).reservedBytes();
}
