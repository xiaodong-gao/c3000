#include "Die2DiePipeline.hpp"

#include "EccAlign.hpp"
#include "IntensityNormalize.h"
#include "PhaseCorrelate.hpp"

#include <chrono>
#include <cstdio>

#include <opencv2/imgproc.hpp>

namespace {

cv::Mat asContinuous(const cv::Mat& src) {
    return src.isContinuous() ? src : src.clone();
}

cv::Mat to8U(const cv::Mat& src) {
    if (src.depth() == CV_8U) {
        return src;
    }
    cv::Mat dst;
    if (src.depth() == CV_16U) {
        src.convertTo(dst, CV_8U, 1.0 / 256.0);
        return dst;
    }
    double minv = 0.0;
    double maxv = 0.0;
    cv::minMaxLoc(src, &minv, &maxv);
    src.convertTo(dst, CV_8U, maxv <= 1.5 ? 255.0 : 1.0);
    return dst;
}

cv::Mat toBgrIfNeeded(const cv::Mat& src) {
    if (src.channels() == 3) {
        return src;
    }
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

bool preparePair(const cv::Mat& src, cv::Mat& original, cv::Mat& gray, const char*& sourceDesc,
                 std::string& error, const char* name) {
    const cv::Mat src8u = to8U(src);
    if (src8u.channels() == 1) {
        original = asContinuous(src8u);
        gray = original;
        sourceDesc = "grayscale (direct)";
        return true;
    }
    if (src8u.channels() == 3) {
        original = asContinuous(src8u);
        cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
        gray = asContinuous(gray);
        sourceDesc = "color BGR -> gray hist / original D2D";
        return true;
    }
    if (src8u.channels() == 4) {
        cv::cvtColor(src8u, original, cv::COLOR_BGRA2BGR);
        original = asContinuous(original);
        cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
        gray = asContinuous(gray);
        sourceDesc = "color BGRA -> gray hist / original D2D";
        return true;
    }
    error = std::string(name) + ": unsupported image channels=" + std::to_string(src.channels());
    return false;
}

struct StageClock {
    bool enabled = false;
    bool syncCuda = false;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    void start() {
        if (!enabled) {
            return;
        }
        if (syncCuda) {
            cudaDeviceSynchronize();
        }
        t0 = std::chrono::steady_clock::now();
    }

    double stop() {
        if (!enabled) {
            return 0.0;
        }
        if (syncCuda) {
            cudaDeviceSynchronize();
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

}  // namespace

const char* pipelineBackendName(PipelineBackend backend) {
    return backend == PipelineBackend::Cuda ? "cuda" : "cpu";
}

const char* pipelineAlignName(PipelineAlign align) {
    return align == PipelineAlign::Ecc ? "ecc" : "phase";
}

const char* pipelineFlowText() {
    return
        "Wafer / paired Die ROI\n"
        "  -> grayscale + histogram threshold  (CPU or CUDA)\n"
        "  -> Die registration (Phase Correlation or ECC, CPU or CUDA)\n"
        "  -> sub-pixel warp of the original data image\n"
        "  -> crop valid common-overlap template/data ROI\n"
        "  -> per-channel intensity normalization (CPU or CUDA)\n"
        "  -> Absolute / Signed / L2 Difference\n"
        "  -> Signed polarity: Bright Mask (diff>+T) / Dark Mask (diff<-T)\n"
        "  -> fused Difference -> Adaptive Threshold\n"
        "  -> Morphology (Open / Close / Dilate / Erode / Remove Noise)\n"
        "  -> Connected Components -> Defect Blob -> Feature Extraction\n"
        "  -> Bright Defect / Dark Defect -> Defect Filtering\n";
}

static PipelineBackend histogramBackendOf(PipelineBackend pipelineBackend, const PipelineContext& ctx) {
    return ctx.overrideHistogramBackend ? ctx.histogramBackend : pipelineBackend;
}

static PipelineBackend normalizeBackendOf(PipelineBackend pipelineBackend, const PipelineContext& ctx) {
    return ctx.overrideNormalizeBackend ? ctx.normalizeBackend : pipelineBackend;
}

PipelineOutput runD2DPipeline(const cv::Mat& templateImg, const cv::Mat& dataImg,
                              PipelineBackend backend, PipelineAlign align, PipelineContext& ctx,
                              bool collectTiming) {
    PipelineOutput out;
    const PipelineBackend histBackend = histogramBackendOf(backend, ctx);
    const PipelineBackend normBackend = normalizeBackendOf(backend, ctx);
    if (!ctx.hist) {
        out.error = "pipeline context is incomplete";
        return out;
    }
    if ((align == PipelineAlign::Phase &&
         ((backend == PipelineBackend::Cpu && !ctx.phase) ||
          (backend == PipelineBackend::Cuda && !ctx.phaseCuda))) ||
        (align == PipelineAlign::Ecc && backend == PipelineBackend::Cuda && !ctx.eccCuda)) {
        out.error = "pipeline context is incomplete";
        return out;
    }
    if (normBackend == PipelineBackend::Cuda && !ctx.intensityCuda) {
        out.error = "pipeline context is incomplete";
        return out;
    }

    StageClock clock;
    clock.enabled = collectTiming;
    clock.syncCuda = backend == PipelineBackend::Cuda || histBackend == PipelineBackend::Cuda ||
                     normBackend == PipelineBackend::Cuda;
    const auto total0 = std::chrono::steady_clock::now();

    if (!preparePair(templateImg, out.templateOrig, out.templateGray, out.templateSource, out.error,
                     "template") ||
        !preparePair(dataImg, out.dataOrig, out.dataGray, out.dataSource, out.error, "data")) {
        return out;
    }
    if (out.templateGray.size() != out.dataGray.size()) {
        out.error = "template/data size mismatch";
        return out;
    }
    if (out.templateOrig.channels() != out.dataOrig.channels()) {
        out.templateOrig = toBgrIfNeeded(out.templateOrig);
        out.dataOrig = toBgrIfNeeded(out.dataOrig);
    }

    const int w = out.templateGray.cols;
    const int h = out.templateGray.rows;
    const int n = w * h;
    out.templateBin.create(h, w, CV_8UC1);
    out.dataBin.create(h, w, CV_8UC1);

    clock.start();
    if (histBackend == PipelineBackend::Cpu) {
        if (!ctx.hist->computeFromGrayCpu(out.templateGray.data, w, h, out.templateHist) ||
            !ctx.hist->computeFromGrayCpu(out.dataGray.data, w, h, out.dataHist) ||
            !ctx.hist->applyCpu(out.templateGray.data, n, out.templateHist.threshold,
                                out.templateBin.data) ||
            !ctx.hist->applyCpu(out.dataGray.data, n, out.dataHist.threshold, out.dataBin.data)) {
            out.error = "histogram CPU failed";
            return out;
        }
    } else {
        const cudaError_t histStatus =
            ctx.hist->computeFromGray(out.templateGray.data, w, h, out.templateHist);
        const cudaError_t dataStatus =
            ctx.hist->computeFromGray(out.dataGray.data, w, h, out.dataHist);
        const cudaError_t applyT = ctx.hist->apply(out.templateGray.data, n,
                                                   out.templateHist.threshold, out.templateBin.data);
        const cudaError_t applyD =
            ctx.hist->apply(out.dataGray.data, n, out.dataHist.threshold, out.dataBin.data);
        if (histStatus != cudaSuccess || dataStatus != cudaSuccess || applyT != cudaSuccess ||
            applyD != cudaSuccess) {
            out.error = "histogram CUDA failed";
            return out;
        }
    }
    out.timing.histMs = clock.stop();
    out.templateT = out.templateHist.threshold;
    out.dataT = out.dataHist.threshold;
    out.fusedT = HistogramThreshold::fuse(out.templateT, out.dataT);

    clock.start();
    if (align == PipelineAlign::Ecc) {
        out.metricName = "cc";
        // ECC 用灰度图（与 OpenCV findTransformECC 示例一致）。二值图梯度稀疏，欧氏模型易 NaN。
        if (backend == PipelineBackend::Cpu) {
            const EccAlignResult ecc = eccShift(out.templateGray, out.dataGray, ctx.eccParams);
            if (!ecc.ok) {
                out.error = ecc.error.empty() ? "ECC CPU did not converge" : ecc.error;
                return out;
            }
            out.shift = ecc.shift;
            out.response = ecc.cc;
        } else {
            EccCudaResult ecc{};
            const cudaError_t status =
                ctx.eccCuda->compute(out.templateGray.data, out.dataGray.data, w, h, ecc);
            if (status != cudaSuccess) {
                out.error = cudaGetErrorString(status);
                return out;
            }
            if (!ecc.ok) {
                out.error = "ECC CUDA did not converge";
                return out;
            }
            out.shift = cv::Point2d(ecc.dx, ecc.dy);
            out.response = ecc.cc;
        }
    } else {
        out.metricName = "response";
        if (backend == PipelineBackend::Cpu) {
            out.shift = ctx.phase->compute(out.templateBin, out.dataBin, &out.response);
        } else {
            PhaseCorrelateResult phase{};
            const cudaError_t status =
                ctx.phaseCuda->compute(out.templateBin.data, out.dataBin.data, w, h, phase);
            if (status != cudaSuccess) {
                out.error = cudaGetErrorString(status);
                return out;
            }
            out.shift = cv::Point2d(phase.dx, phase.dy);
            out.response = phase.response;
        }
    }
    out.timing.alignMs = clock.stop();

    out.dataBinAligned = shiftImage(out.dataBin, out.shift, cv::INTER_NEAREST);
    out.dataOrigAligned = shiftImage(out.dataOrig, out.shift, cv::INTER_LINEAR);
    out.commonRoi = commonOverlapRoi(out.templateOrig.size(), out.shift);
    if (out.commonRoi.width < 16 || out.commonRoi.height < 16) {
        out.error = "common overlap is too small for defect detection";
        return out;
    }
    out.templateCommon = out.templateOrig(out.commonRoi).clone();
    out.dataCommon = out.dataOrigAligned(out.commonRoi).clone();

    // Radiometric normalization must operate only on the valid, aligned overlap. This prevents
    // the warp border from biasing the intensity statistics and the subsequent D2D difference.
    clock.start();
    const IntensityNormalizeOutput normalized =
        normBackend == PipelineBackend::Cpu
            ? intensityNormalize(out.templateCommon, out.dataCommon)
            : intensityNormalizeCuda(*ctx.intensityCuda, out.templateCommon, out.dataCommon);
    if (!normalized.ok) {
        out.error = normalized.error.empty() ? "intensity normalization failed" : normalized.error;
        return out;
    }
    out.templateNormalized = normalized.templateNormalized;
    out.dataNormalized = normalized.dataNormalized;
    out.absoluteDifference = normalized.absoluteDifference;
    out.l2Difference = normalized.l2Difference;
    out.normalizeStats = normalized.stats;
    out.timing.normalizeDiffMs = clock.stop();

    out.signedDifference = {};
    applyDefectPolarity(out.l2Difference, out.templateNormalized, out.dataNormalized,
                        ctx.defectParams, out.signedDifference, out.brightMask, out.darkMask,
                        out.polarityMask, out.fusedDifference, out.polarityT, out.polarityDarkT);

    clock.start();
    out.adaptiveMask = adaptiveThresholdDiff(out.fusedDifference, ctx.defectParams.adaptive);
    if (out.adaptiveMask.empty()) {
        out.error = "adaptive threshold failed";
        return out;
    }
    out.timing.adaptiveMs = clock.stop();

    clock.start();
    out.morphMask =
        mergeMorphWithDarkChannel(out.adaptiveMask, out.darkMask, ctx.defectParams.morph);
    out.timing.morphMs = clock.stop();

    clock.start();
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(out.morphMask, labels, stats, centroids, 8, CV_32S);
    const std::vector<DefectCandidate> raw = extractDefectCandidates(
        out.morphMask, out.l2Difference, out.signedDifference, out.brightMask, out.darkMask);
    out.candidates =
        filterDefects(raw, ctx.defectParams.blob, ctx.defectParams.filter, out.morphMask.size());
    out.blobMask = maskFromCandidates(out.morphMask.size(), out.candidates, labels);
    out.timing.blobMs = clock.stop();

    clock.start();
    out.defectOverlay = drawDefectOverlay(out.templateNormalized, out.candidates);
    out.timing.candidateMs = clock.stop();

    if (collectTiming) {
        if (backend == PipelineBackend::Cuda || histBackend == PipelineBackend::Cuda ||
            normBackend == PipelineBackend::Cuda) {
            cudaDeviceSynchronize();
        }
        out.timing.totalMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total0)
                .count();
    }
    out.ok = true;
    return out;
}

PipelineOutput runAlignPipeline(const cv::Mat& templateImg, const cv::Mat& dataImg,
                                PipelineBackend backend, PipelineAlign align, PipelineContext& ctx,
                                bool collectTiming) {
    return runD2DPipeline(templateImg, dataImg, backend, align, ctx, collectTiming);
}
