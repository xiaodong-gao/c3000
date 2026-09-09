#pragma once

#include "DefectDetect.hpp"
#include "EccAlign.hpp"
#include "EccAlign.cuh"
#include "HistogramThreshold.cuh"
#include "IntensityNormalize.cuh"
#include "PhaseCorrelate.cuh"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

enum class PipelineBackend { Cpu, Cuda };
enum class PipelineAlign { Phase, Ecc };

struct PipelineTiming {
    double histMs = 0.0;
    double alignMs = 0.0;
    double normalizeDiffMs = 0.0;
    double adaptiveMs = 0.0;
    double morphMs = 0.0;
    double blobMs = 0.0;
    double candidateMs = 0.0;
    double totalMs = 0.0;
};

struct PipelineOutput {
    bool ok = false;
    std::string error;

    cv::Mat templateGray;
    cv::Mat dataGray;
    cv::Mat templateOrig;
    cv::Mat dataOrig;
    cv::Mat templateBin;
    cv::Mat dataBin;
    cv::Mat dataBinAligned;
    cv::Mat dataOrigAligned;
    cv::Mat templateCommon;
    cv::Mat dataCommon;
    cv::Mat templateNormalized;
    cv::Mat dataNormalized;
    cv::Mat absoluteDifference;
    cv::Mat l2Difference;
    cv::Mat adaptiveMask;
    cv::Mat morphMask;
    cv::Mat blobMask;
    cv::Mat defectOverlay;
    std::vector<DefectCandidate> candidates;

    HistogramThresholdResult templateHist{};
    HistogramThresholdResult dataHist{};
    int templateT = 0;
    int dataT = 0;
    int fusedT = 0;

    cv::Point2d shift;
    double response = 0.0;
    const char* metricName = "response";
    cv::Rect commonRoi;
    IntensityNormalizeStats normalizeStats;

    const char* templateSource = "grayscale";
    const char* dataSource = "grayscale";
    PipelineTiming timing;
};

struct PipelineContext {
    HistogramThreshold* hist = nullptr;
    PhaseCorrelateCuda* phaseCuda = nullptr;
    EccAlignCuda* eccCuda = nullptr;
    IntensityNormalizeCuda* intensityCuda = nullptr;
    DefectDetectParams defectParams;
    EccAlignParams eccParams;

    // When false, histogram / normalize follow runD2DPipeline's backend argument.
    bool overrideHistogramBackend = false;
    PipelineBackend histogramBackend = PipelineBackend::Cpu;
    bool overrideNormalizeBackend = false;
    PipelineBackend normalizeBackend = PipelineBackend::Cpu;
};

const char* pipelineBackendName(PipelineBackend backend);
const char* pipelineAlignName(PipelineAlign align);
const char* pipelineFlowText();

// Registration -> ROI -> normalization -> difference -> defect candidates.
PipelineOutput runD2DPipeline(const cv::Mat& templateImg, const cv::Mat& dataImg,
                              PipelineBackend backend, PipelineAlign align, PipelineContext& ctx,
                              bool collectTiming = false);

PipelineOutput runAlignPipeline(const cv::Mat& templateImg, const cv::Mat& dataImg,
                                PipelineBackend backend, PipelineAlign align, PipelineContext& ctx,
                                bool collectTiming = false);
