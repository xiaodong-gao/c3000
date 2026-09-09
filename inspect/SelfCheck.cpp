#include "SelfCheck.hpp"

#include "die_cli.h"
#include "die_inspect.h"

#include "DefectDetect.hpp"
#include "EccAlign.cuh"
#include "EccAlign.hpp"
#include "HistogramThreshold.cuh"
#include "IntensityNormalize.cuh"
#include "IntensityNormalize.h"
#include "Mssim.cuh"
#include "Mssim.hpp"
#include "PhaseCorrelate.cuh"
#include "PhaseCorrelate.hpp"
#include "Pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct TestStats {
    int passed = 0;
    int failed = 0;
};

void check(TestStats& stats, bool ok, const std::string& name, const std::string& detail);

cv::Mat makePattern(int width, int height, bool binary) {
    cv::Mat img(height, width, CV_8UC1, cv::Scalar(binary ? 0 : 32));
    for (int y = 36; y < height - 36; y += 44) {
        for (int x = 36; x < width - 36; x += 44) {
            cv::circle(img, cv::Point(x, y), 10, cv::Scalar(binary ? 255 : 210),
                       binary ? cv::FILLED : 3);
        }
    }
    cv::rectangle(img, cv::Point(width / 6, height / 5), cv::Point(width / 2, height / 2),
                  cv::Scalar(255), binary ? cv::FILLED : 2);
    cv::rectangle(img, cv::Point(width * 3 / 5, height * 3 / 5),
                  cv::Point(width * 5 / 6, height * 4 / 5), cv::Scalar(binary ? 255 : 180),
                  cv::FILLED);
    return img;
}

cv::Mat shifted(const cv::Mat& src, const cv::Point2d& shift, int interpolation) {
    return shiftImage(src, shift, interpolation);
}

double medianMs(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    if (n == 0) {
        return 0.0;
    }
    if (n % 2 == 1) {
        return samples[n / 2];
    }
    return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

template <typename Fn>
double benchmarkMedian(int iters, bool synchronizeCuda, Fn&& fn) {
    fn();
    if (synchronizeCuda) {
        const cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        if (synchronizeCuda) {
            const cudaError_t status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(status));
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return medianMs(samples);
}

std::string environmentValue(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value) {
        return {};
    }
    const std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    return value ? value : "";
#endif
}

void testCommandLine(TestStats& stats) {
    const cutdie::CommandLine empty = cutdie::parse_command_line({});
    check(stats, empty.mode == cutdie::RunMode::Help, "cli defaults to help",
          "empty arguments did not select help");

    const cutdie::CommandLine bench =
        cutdie::parse_command_line({"--benchmark", "12"});
    check(stats, bench.mode == cutdie::RunMode::Benchmark && bench.benchmark_iterations == 12,
          "cli benchmark mode", "mode/iterations mismatch");

    const cutdie::CommandLine process = cutdie::parse_command_line(
        {"--dir", "samples", "--resolution-um", "25", "--align", "ecc-cuda",
         "--histogram", "auto", "--normalize", "cuda", "--template-neighbor", "3",
         "--recursive"});
    check(stats,
          process.mode == cutdie::RunMode::Process && process.options.image.empty() &&
              process.options.xml.empty() && process.options.dir == "samples" &&
              process.options.resolution_um == 25.0 &&
              process.options.align_method == cutdie::AlignMethod::EccCuda &&
              process.options.histogram_method == cutdie::HistogramMethod::Auto &&
              process.options.normalize_method == cutdie::NormalizeMethod::Cuda &&
              process.options.template_neighbor && process.options.template_neighbor_min == 3 &&
              process.options.recursive,
          "cli processing overrides", "parsed options mismatch");

    bool rejected = false;
    try {
        (void)cutdie::parse_command_line({"--benchmark", "0"});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(stats, rejected, "cli rejects invalid iterations", "--benchmark 0 was accepted");

    rejected = false;
    try {
        (void)cutdie::parse_command_line({"--dir", "samples", "--resolution-um", "inf"});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(stats, rejected, "cli rejects non-finite resolution", "infinite resolution was accepted");
}

void check(TestStats& stats, bool ok, const std::string& name, const std::string& detail) {
    if (ok) {
        ++stats.passed;
        printf("  PASS  %s\n", name.c_str());
        return;
    }
    ++stats.failed;
    printf("  FAIL  %s  %s\n", name.c_str(), detail.c_str());
}

void checkNear(TestStats& stats, double actual, double expected, double tol, const std::string& name) {
    const bool ok = std::abs(actual - expected) <= tol;
    char detail[128];
    std::snprintf(detail, sizeof(detail), "actual=%.4f expected=%.4f tol=%.4f", actual, expected, tol);
    check(stats, ok, name, detail);
}

void checkShift(TestStats& stats, const cv::Point2d& actual, const cv::Point2d& expected, double tol,
                const std::string& name) {
    const double err = std::hypot(actual.x - expected.x, actual.y - expected.y);
    char detail[160];
    std::snprintf(detail, sizeof(detail), "got=(%.4f, %.4f) expect=(%.4f, %.4f) err=%.4f tol=%.4f",
                  actual.x, actual.y, expected.x, expected.y, err, tol);
    check(stats, err <= tol, name, detail);
}

bool cudaOk(cudaError_t status, const char* what, TestStats* stats = nullptr) {
    if (status == cudaSuccess) {
        return true;
    }
    if (stats) {
        check(*stats, false, what, cudaGetErrorString(status));
    } else {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    }
    return false;
}

cv::Point2d cudaShift(PhaseCorrelateCuda& engine, const cv::Mat& tmpl, const cv::Mat& data,
                      double* response, TestStats& stats, const char* name) {
    PhaseCorrelateResult result{};
    if (!cudaOk(engine.compute(tmpl.data, data.data, tmpl.cols, tmpl.rows, result), name, &stats)) {
        return cv::Point2d(1e9, 1e9);
    }
    if (response) {
        *response = result.response;
    }
    return cv::Point2d(result.dx, result.dy);
}

double overlapMae(const cv::Mat& tmpl, const cv::Mat& data, const cv::Point2d& shift,
                  int interpolation) {
    const cv::Mat aligned = shifted(data, shift, interpolation);
    const cv::Rect roi = commonOverlapRoi(tmpl.size(), shift);
    if (roi.width < 16 || roi.height < 16) {
        return 1e9;
    }
    cv::Mat diff;
    cv::absdiff(tmpl(roi), aligned(roi), diff);
    return cv::mean(diff)[0];
}

void checkEquivalentShift(TestStats& stats, const cv::Point2d& actual, const cv::Point2d& expected,
                          double tol, const std::string& name) {
    const double same = std::hypot(actual.x - expected.x, actual.y - expected.y);
    const double opposite = std::hypot(actual.x + expected.x, actual.y + expected.y);
    const double err = std::min(same, opposite);
    char detail[192];
    std::snprintf(detail, sizeof(detail),
                  "got=(%.4f, %.4f) expect=±(%.4f, %.4f) err=%.4f tol=%.4f", actual.x, actual.y,
                  expected.x, expected.y, err, tol);
    check(stats, err <= tol, name, detail);
}

void testPhaseZeroShift(TestStats& stats, PhaseCorrelateCuda& cuda) {
    const cv::Mat tmpl = makePattern(256, 256, true);
    double cpuResp = 0.0;
    double gpuResp = 0.0;
    const cv::Point2d cpu = phaseShiftBinary(tmpl, tmpl, &cpuResp);
    const cv::Point2d gpu = cudaShift(cuda, tmpl, tmpl, &gpuResp, stats, "phase-cuda zero");
    checkShift(stats, cpu, cv::Point2d(0, 0), 0.15, "phase-cpu zero shift");
    checkShift(stats, gpu, cv::Point2d(0, 0), 0.15, "phase-cuda zero shift");
    check(stats, gpuResp > 0.0, "phase-cuda zero response", "response <= 0");
}

void testPhaseKnownShifts(TestStats& stats, PhaseCorrelateCuda& cuda) {
    const cv::Mat tmpl = makePattern(256, 256, true);
    const cv::Point2d cases[] = {
        {3.0, 1.0}, {1.0, -4.0}, {4.0, 3.0}, {-2.0, 3.0}, {3.0, -1.0},
    };
    for (const cv::Point2d& gt : cases) {
        const cv::Mat data = shifted(tmpl, gt, cv::INTER_NEAREST);
        const cv::Point2d cpu = phaseShiftBinary(tmpl, data, nullptr);
        const cv::Point2d gpu = cudaShift(cuda, tmpl, data, nullptr, stats, "phase-cuda known");
        char magName[80];
        char alignCpu[80];
        char alignGpu[80];
        char agreeName[80];
        std::snprintf(magName, sizeof(magName), "phase |shift| (%.0f,%.0f)", gt.x, gt.y);
        std::snprintf(alignCpu, sizeof(alignCpu), "phase-cpu aligns (%.0f,%.0f)", gt.x, gt.y);
        std::snprintf(alignGpu, sizeof(alignGpu), "phase-cuda aligns (%.0f,%.0f)", gt.x, gt.y);
        std::snprintf(agreeName, sizeof(agreeName), "phase cpu vs cuda (%.0f,%.0f)", gt.x, gt.y);

        checkNear(stats, std::hypot(gpu.x, gpu.y), std::hypot(gt.x, gt.y), 0.30, magName);
        check(stats, overlapMae(tmpl, data, cpu, cv::INTER_NEAREST) <= 8.0, alignCpu,
              "mae=" + std::to_string(overlapMae(tmpl, data, cpu, cv::INTER_NEAREST)));
        check(stats, overlapMae(tmpl, data, gpu, cv::INTER_NEAREST) <= 8.0, alignGpu,
              "mae=" + std::to_string(overlapMae(tmpl, data, gpu, cv::INTER_NEAREST)));
        checkEquivalentShift(stats, gpu, cpu, 0.25, agreeName);
    }
}

void testPhaseSubpixel(TestStats& stats, PhaseCorrelateCuda& cuda) {
    cv::Mat tmpl(320, 320, CV_32F, cv::Scalar(20));
    for (int y = 48; y < 272; y += 56) {
        for (int x = 48; x < 272; x += 56) {
            cv::circle(tmpl, cv::Point(x, y), 16, cv::Scalar(220), cv::FILLED);
        }
    }
    cv::GaussianBlur(tmpl, tmpl, cv::Size(11, 11), 2.5);
    cv::Mat tmpl8;
    tmpl.convertTo(tmpl8, CV_8U);

    const cv::Point2d gt(2.40, -1.60);
    const cv::Mat data = shifted(tmpl8, gt, cv::INTER_LINEAR);
    const cv::Point2d cpu = phaseShiftBinary(tmpl8, data, nullptr);
    const cv::Point2d gpu = cudaShift(cuda, tmpl8, data, nullptr, stats, "phase-cuda subpixel");
    checkEquivalentShift(stats, gpu, cpu, 0.20, "phase cpu vs cuda subpixel");
    checkNear(stats, std::hypot(cpu.x, cpu.y), std::hypot(gt.x, gt.y), 0.45,
              "phase-cpu |subpixel| (2.40,-1.60)");
    checkNear(stats, std::hypot(gpu.x, gpu.y), std::hypot(gt.x, gt.y), 0.45,
              "phase-cuda |subpixel| (2.40,-1.60)");
}

void testPhaseLargeShifts(TestStats& stats, PhaseCorrelateCuda& cuda) {
    cv::Mat tmpl(256, 256, CV_8UC1);
    cv::RNG rng(0xC3000);
    rng.fill(tmpl, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(tmpl, tmpl, cv::Size(5, 5), 1.0);

    const cv::Point2d cases[] = {
        {20.0, -17.0}, {-23.0, 18.0}, {19.0, 22.0}, {-21.0, -20.0},
    };
    for (const cv::Point2d& gt : cases) {
        const cv::Mat data = shifted(tmpl, gt, cv::INTER_LINEAR);
        double cpuResponse = 0.0;
        double gpuResponse = 0.0;
        const cv::Point2d cpu = phaseShiftBinary(tmpl, data, &cpuResponse);
        const cv::Point2d gpu =
            cudaShift(cuda, tmpl, data, &gpuResponse, stats, "phase-cuda large shift");
        const cv::Point2d expected(-gt.x, -gt.y);
        const std::string suffix = " (" + std::to_string(static_cast<int>(gt.x)) + "," +
                                   std::to_string(static_cast<int>(gt.y)) + ")";
        checkShift(stats, cpu, expected, 0.65, "phase-cpu large shift" + suffix);
        checkShift(stats, gpu, expected, 0.65, "phase-cuda large shift" + suffix);
        checkShift(stats, gpu, cpu, 0.30, "phase cpu vs cuda large shift" + suffix);
        check(stats, cpuResponse > 0.0 && gpuResponse > 0.0,
              "phase large-shift response" + suffix, "response <= 0");
    }
}

void testEccShift(TestStats& stats, EccAlignCuda& eccCuda) {
    const cv::Mat tmpl = makePattern(256, 256, false);
    const cv::Point2d cases[] = {{3.0, -2.0}, {1.5, 2.5}, {-2.0, 1.0}};
    for (const cv::Point2d& gt : cases) {
        const cv::Mat data = shifted(tmpl, gt, cv::INTER_LINEAR);
        const EccAlignResult cpu = eccShift(tmpl, data);
        EccCudaResult gpu{};
        const bool gpuOk =
            cudaOk(eccCuda.compute(tmpl.data, data.data, tmpl.cols, tmpl.rows, gpu),
                   "ecc-cuda compute", &stats) &&
            gpu.ok;

        char cpuName[64];
        char gpuName[64];
        char agreeName[64];
        std::snprintf(cpuName, sizeof(cpuName), "ecc-cpu recover (%.1f,%.1f)", gt.x, gt.y);
        std::snprintf(gpuName, sizeof(gpuName), "ecc-cuda recover (%.1f,%.1f)", gt.x, gt.y);
        std::snprintf(agreeName, sizeof(agreeName), "ecc cpu vs cuda (%.1f,%.1f)", gt.x, gt.y);

        check(stats, cpu.ok, "ecc-cpu converge", cpu.error);
        if (cpu.ok) {
            checkEquivalentShift(stats, cpu.shift, gt, 0.35, cpuName);
            check(stats, overlapMae(tmpl, data, cpu.shift, cv::INTER_LINEAR) <= 8.0,
                  std::string("ecc-cpu aligns (") + std::to_string(gt.x) + "," +
                      std::to_string(gt.y) + ")",
                  "mae=" + std::to_string(overlapMae(tmpl, data, cpu.shift, cv::INTER_LINEAR)));
        }
        check(stats, gpuOk, "ecc-cuda converge", gpuOk ? "" : "did not converge");
        if (gpuOk) {
            checkEquivalentShift(stats, cv::Point2d(gpu.dx, gpu.dy), gt, 0.35, gpuName);
            check(stats, overlapMae(tmpl, data, cv::Point2d(gpu.dx, gpu.dy), cv::INTER_LINEAR) <= 8.0,
                  std::string("ecc-cuda aligns (") + std::to_string(gt.x) + "," +
                      std::to_string(gt.y) + ")",
                  "mae=" +
                      std::to_string(overlapMae(tmpl, data, cv::Point2d(gpu.dx, gpu.dy),
                                                cv::INTER_LINEAR)));
        }
        if (cpu.ok && gpuOk) {
            checkShift(stats, cv::Point2d(gpu.dx, gpu.dy), cpu.shift, 0.20, agreeName);
            char ccName[64];
            std::snprintf(ccName, sizeof(ccName), "ecc cc cpu vs cuda (%.1f,%.1f)", gt.x, gt.y);
            checkNear(stats, gpu.cc, cpu.cc, 0.03, ccName);
        }
    }

    const EccAlignResult zeroCpu = eccShift(tmpl, tmpl);
    EccCudaResult zeroGpu{};
    if (cudaOk(eccCuda.compute(tmpl.data, tmpl.data, tmpl.cols, tmpl.rows, zeroGpu),
               "ecc-cuda zero", &stats) &&
        zeroGpu.ok && zeroCpu.ok) {
        checkShift(stats, cv::Point2d(zeroGpu.dx, zeroGpu.dy), cv::Point2d(0, 0), 0.15,
                   "ecc-cuda zero shift");
    }
}

void testEccWorkspaceIsolation(TestStats& stats) {
    EccCudaParams p3 = eccCudaDefaultParams();
    p3.maxIters = 20;
    p3.gaussFiltSize = 3;
    EccCudaParams p7 = p3;
    p7.gaussFiltSize = 7;
    EccAlignCuda ecc3(p3);
    EccAlignCuda ecc7(p7);

    const cv::Mat tmpl = makePattern(256, 256, false);
    const cv::Mat data = shifted(tmpl, cv::Point2d(2.5, -1.5), cv::INTER_LINEAR);
    EccCudaResult first{};
    EccCudaResult middle{};
    EccCudaResult second{};
    const cudaError_t s1 = ecc3.compute(tmpl.data, data.data, tmpl.cols, tmpl.rows, first);
    const cudaError_t s2 = ecc7.compute(tmpl.data, data.data, tmpl.cols, tmpl.rows, middle);
    const cudaError_t s3 = ecc3.compute(tmpl.data, data.data, tmpl.cols, tmpl.rows, second);
    check(stats,
          s1 == cudaSuccess && s2 == cudaSuccess && s3 == cudaSuccess && first.ok && middle.ok &&
              second.ok,
          "ecc independent workspaces run", "one of the alternating calls failed");
    if (s1 == cudaSuccess && s3 == cudaSuccess && first.ok && second.ok) {
        checkShift(stats, cv::Point2d(second.dx, second.dy), cv::Point2d(first.dx, first.dy),
                   1e-5, "ecc gaussian workspace isolation");
        checkNear(stats, second.cc, first.cc, 1e-8, "ecc workspace cc stability");
    }
}

void testHistogramRules(TestStats& stats, HistogramThreshold& hist) {
    cv::Mat twoPeak(240, 320, CV_8UC1);
    twoPeak.colRange(0, 160).setTo(40);
    twoPeak.colRange(160, 320).setTo(200);
    HistogramThresholdResult two{};
    if (cudaOk(hist.computeFromGray(twoPeak.data, twoPeak.cols, twoPeak.rows, two),
               "hist two-peak", &stats)) {
        check(stats, two.peakCount == 2, "hist two-peak count",
              "peakCount=" + std::to_string(two.peakCount));
        check(stats, two.threshold >= 90 && two.threshold <= 150, "hist two-peak T in valley",
              "T=" + std::to_string(two.threshold));
        HistogramThresholdResult twoCpu{};
        if (hist.computeFromGrayCpu(twoPeak.data, twoPeak.cols, twoPeak.rows, twoCpu)) {
            checkNear(stats, twoCpu.threshold, two.threshold, 0.0, "hist cpu vs cuda T");
            check(stats, twoCpu.peakCount == two.peakCount, "hist cpu vs cuda peaks",
                  "cpu=" + std::to_string(twoCpu.peakCount) +
                      " cuda=" + std::to_string(two.peakCount));
        } else {
            check(stats, false, "hist cpu two-peak", "computeFromGrayCpu failed");
        }
    }

    cv::Mat onePeak(200, 200, CV_8UC1, cv::Scalar(80));
    HistogramThresholdResult one{};
    if (cudaOk(hist.computeFromGray(onePeak.data, onePeak.cols, onePeak.rows, one),
               "hist one-peak", &stats)) {
        check(stats, one.peakCount == 1, "hist one-peak count",
              "peakCount=" + std::to_string(one.peakCount));
        checkNear(stats, one.threshold, (one.filteredPeaks[0].gray + 255) / 2, 1.0,
                  "hist one-peak rule");
    }

    checkNear(stats, HistogramThreshold::fuse(10, 20), 15.0, 0.0, "hist fuse");
}

void testHistogramApplyAndColor(TestStats& stats, HistogramThreshold& hist) {
    cv::Mat gray(180, 220, CV_8UC1);
    gray.rowRange(0, 90).setTo(30);
    gray.rowRange(90, 180).setTo(210);
    HistogramThresholdResult grayResult{};
    if (!cudaOk(hist.computeFromGray(gray.data, gray.cols, gray.rows, grayResult),
                "hist gray", &stats)) {
        return;
    }

    cv::Mat binary(gray.size(), CV_8UC1);
    if (cudaOk(hist.apply(gray.data, gray.rows * gray.cols, grayResult.threshold, binary.data),
               "hist apply", &stats)) {
        double minv = 0.0;
        double maxv = 0.0;
        cv::minMaxLoc(binary, &minv, &maxv);
        check(stats, minv == 0.0 && maxv == 255.0, "hist apply binary range", "not 0/255");
    }

    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    HistogramThresholdResult colorResult{};
    cv::Mat grayOut(gray.size(), CV_8UC1);
    if (cudaOk(hist.computeFromBgr(bgr.data, bgr.cols, bgr.rows, colorResult, grayOut.data),
               "hist color->gray", &stats)) {
        checkNear(stats, colorResult.threshold, grayResult.threshold, 1.0,
                  "hist gray vs BGR-of-gray T");
        check(stats, cv::countNonZero(grayOut != gray) == 0, "hist BGR-of-gray equals gray",
              "converted gray mismatch");
    }
}

double mapMae(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::mean(diff)[0];
}

double maxAbsError(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.size() != b.size() || a.type() != b.type()) {
        return 1e9;
    }
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    double maxValue = 0.0;
    cv::minMaxLoc(diff.reshape(1), nullptr, &maxValue);
    return maxValue;
}

void testIntensityNormalize(TestStats& stats, IntensityNormalizeCuda& intensityCuda) {
    cv::Mat reference(96, 128, CV_8UC1);
    for (int y = 0; y < reference.rows; ++y) {
        for (int x = 0; x < reference.cols; ++x) {
            reference.at<unsigned char>(y, x) =
                static_cast<unsigned char>(30 + ((x * 7 + y * 11) % 171));
        }
    }
    cv::Mat source;
    reference.convertTo(source, CV_8U, 0.70, 25.0);

    const IntensityNormalizeOutput cpu = intensityNormalize(reference, source);
    const IntensityNormalizeOutput gpu = intensityNormalizeCuda(intensityCuda, reference, source);
    check(stats, cpu.ok, "normalize cpu gray", cpu.error);
    check(stats, gpu.ok, "normalize cuda gray", gpu.error);
    if (cpu.ok && gpu.ok) {
        check(stats, cv::mean(cpu.absoluteDifference)[0] <= 0.7,
              "normalize removes gray gain/offset",
              "mae=" + std::to_string(cv::mean(cpu.absoluteDifference)[0]));
        check(stats, mapMae(cpu.dataNormalized, gpu.dataNormalized) <= 1.0,
              "normalize gray cpu vs cuda",
              "mae=" + std::to_string(mapMae(cpu.dataNormalized, gpu.dataNormalized)));
        check(stats, maxAbsError(cpu.dataNormalized, gpu.dataNormalized) <= 1.0,
              "normalize gray cpu vs cuda max error",
              "max=" + std::to_string(maxAbsError(cpu.dataNormalized, gpu.dataNormalized)));
        check(stats, mapMae(cpu.absoluteDifference, gpu.absoluteDifference) <= 1.0,
              "absolute difference cpu vs cuda",
              "mae=" + std::to_string(mapMae(cpu.absoluteDifference, gpu.absoluteDifference)));
        check(stats, mapMae(cpu.l2Difference, gpu.l2Difference) <= 1.0,
              "L2 difference cpu vs cuda",
              "mae=" + std::to_string(mapMae(cpu.l2Difference, gpu.l2Difference)));
        check(stats, maxAbsError(cpu.l2Difference, gpu.l2Difference) <= 1.0,
              "L2 difference cpu vs cuda max error",
              "max=" + std::to_string(maxAbsError(cpu.l2Difference, gpu.l2Difference)));
    }

    std::vector<cv::Mat> referenceChannels{
        reference, 220 - reference / 2, 40 + reference / 3,
    };
    cv::Mat referenceColor;
    cv::merge(referenceChannels, referenceColor);
    std::vector<cv::Mat> sourceChannels(3);
    referenceChannels[0].convertTo(sourceChannels[0], CV_8U, 0.80, 12.0);
    referenceChannels[1].convertTo(sourceChannels[1], CV_8U, 0.65, 30.0);
    referenceChannels[2].convertTo(sourceChannels[2], CV_8U, 1.10, -8.0);
    cv::Mat sourceColor;
    cv::merge(sourceChannels, sourceColor);

    const IntensityNormalizeOutput colorCpu = intensityNormalize(referenceColor, sourceColor);
    const IntensityNormalizeOutput colorGpu =
        intensityNormalizeCuda(intensityCuda, referenceColor, sourceColor);
    check(stats, colorCpu.ok && colorGpu.ok, "normalize color cpu/cuda",
          colorCpu.ok ? colorGpu.error : colorCpu.error);
    if (colorCpu.ok && colorGpu.ok) {
        check(stats, colorCpu.l2Difference.type() == CV_32FC1,
              "normalize color L2 is scalar float", "unexpected L2 type");
        check(stats, mapMae(colorCpu.dataNormalized, colorGpu.dataNormalized) <= 1.0,
              "normalize color cpu vs cuda",
              "mae=" + std::to_string(mapMae(colorCpu.dataNormalized, colorGpu.dataNormalized)));
        check(stats, maxAbsError(colorCpu.dataNormalized, colorGpu.dataNormalized) <= 1.0,
              "normalize color cpu vs cuda max error",
              "max=" +
                  std::to_string(maxAbsError(colorCpu.dataNormalized, colorGpu.dataNormalized)));

        const int px = referenceColor.cols / 2;
        const int py = referenceColor.rows / 2;
        const cv::Vec3b absolute = colorCpu.absoluteDifference.at<cv::Vec3b>(py, px);
        const double expectedL2 =
            std::sqrt(static_cast<double>(absolute[0]) * absolute[0] +
                      static_cast<double>(absolute[1]) * absolute[1] +
                      static_cast<double>(absolute[2]) * absolute[2]);
        checkNear(stats, colorCpu.l2Difference.at<float>(py, px), expectedL2, 1e-5,
                  "normalize color L2 definition");
    }


    const cv::Mat flatTemplate(48, 64, CV_8UC1, cv::Scalar(120));
    const cv::Mat flatData(48, 64, CV_8UC1, cv::Scalar(80));
    const IntensityNormalizeOutput flatCpu = intensityNormalize(flatTemplate, flatData);
    const IntensityNormalizeOutput flatGpu =
        intensityNormalizeCuda(intensityCuda, flatTemplate, flatData);
    check(stats, flatCpu.ok && flatGpu.ok, "normalize flat ROI cpu/cuda",
          flatCpu.ok ? flatGpu.error : flatCpu.error);
    if (flatCpu.ok && flatGpu.ok) {
        check(stats, cv::countNonZero(flatCpu.absoluteDifference) == 0 &&
                         cv::countNonZero(flatGpu.absoluteDifference) == 0,
              "normalize flat ROI offset fallback", "flat ROI difference is non-zero");
        checkNear(stats, flatGpu.stats.gain[0], 1.0, 1e-12, "normalize flat ROI gain");
        checkNear(stats, flatGpu.stats.bias[0], 40.0, 1e-12, "normalize flat ROI bias");
    }

    cv::Mat robustRef(128, 160, CV_8UC1);
    for (int y = 0; y < robustRef.rows; ++y) {
        for (int x = 0; x < robustRef.cols; ++x) {
            robustRef.at<unsigned char>(y, x) =
                static_cast<unsigned char>(150 + ((x * 3 + y * 5) % 41));
        }
    }
    cv::Mat robustSrc = robustRef.clone();
    const cv::Rect block(24, 20, 72, 72);
    cv::rectangle(robustSrc, block, cv::Scalar(255), cv::FILLED);
    const IntensityNormalizeOutput robustOut = intensityNormalize(robustRef, robustSrc);
    const IntensityNormalizeOutput robustGpu = intensityNormalizeCuda(intensityCuda, robustRef, robustSrc);
    check(stats, robustOut.ok, "normalize large-block defect ok", robustOut.error);
    check(stats, robustGpu.ok, "normalize large-block defect cuda ok", robustGpu.error);
    if (robustOut.ok) {
        const cv::Scalar inner = cv::mean(robustOut.l2Difference(block));
        check(stats, inner[0] >= 25.0, "normalize keeps large-block L2",
              "meanL2=" + std::to_string(inner[0]));
    }
    if (robustOut.ok && robustGpu.ok) {
        check(stats, mapMae(robustOut.dataNormalized, robustGpu.dataNormalized) <= 1.0,
              "normalize large-block cpu vs cuda",
              "mae=" + std::to_string(mapMae(robustOut.dataNormalized, robustGpu.dataNormalized)));
        const cv::Scalar innerGpu = cv::mean(robustGpu.l2Difference(block));
        check(stats, innerGpu[0] >= 25.0, "normalize cuda keeps large-block L2",
              "meanL2=" + std::to_string(innerGpu[0]));
    }
}

void testMssimAndOverlap(TestStats& stats, MssimCuda& mssimCuda) {
    const cv::Mat a = makePattern(192, 160, false);
    const MssimResult sameCpu = computeMssim(a, a);
    const MssimResult sameGpu = computeMssimCuda(mssimCuda, a, a);
    check(stats, sameCpu.mssim > 0.98, "mssim-cpu identical gray",
          "MSSIM=" + std::to_string(sameCpu.mssim));
    check(stats, sameGpu.mssim > 0.98, "mssim-cuda identical gray",
          "MSSIM=" + std::to_string(sameGpu.mssim));
    checkNear(stats, sameGpu.mssim, sameCpu.mssim, 5e-3, "mssim cpu vs cuda identical gray");
    check(stats, mapMae(sameCpu.map, sameGpu.map) <= 0.02, "mssim map mae identical gray",
          "mae=" + std::to_string(mapMae(sameCpu.map, sameGpu.map)));

    cv::Mat color;
    cv::cvtColor(a, color, cv::COLOR_GRAY2BGR);
    const MssimResult colorCpu = computeMssim(color, color);
    const MssimResult colorGpu = computeMssimCuda(mssimCuda, color, color);
    check(stats, colorGpu.mssim > 0.98, "mssim-cuda identical color",
          "MSSIM=" + std::to_string(colorGpu.mssim));
    checkNear(stats, colorGpu.mssim, colorCpu.mssim, 5e-3, "mssim cpu vs cuda identical color");

    cv::Mat b = a.clone();
    cv::circle(b, cv::Point(80, 70), 18, cv::Scalar(10), cv::FILLED);
    const MssimResult diffCpu = computeMssim(a, b);
    const MssimResult diffGpu = computeMssimCuda(mssimCuda, a, b);
    check(stats, diffCpu.mssim < sameCpu.mssim - 0.01, "mssim-cpu drops on change",
          "MSSIM=" + std::to_string(diffCpu.mssim));
    checkNear(stats, diffGpu.mssim, diffCpu.mssim, 2e-2, "mssim cpu vs cuda changed gray");

    const cv::Mat fusionCpu = fuseByMssim(a, b, diffCpu.map);
    const cv::Mat fusionGpu = fuseByMssimCuda(mssimCuda, a, b, diffGpu.map);
    check(stats, fusionGpu.size() == a.size() && fusionGpu.type() == a.type(), "mssim-cuda fusion size",
          "unexpected fusion output");
    cv::Mat fusionDiff;
    cv::absdiff(fusionCpu, fusionGpu, fusionDiff);
    check(stats, cv::mean(fusionDiff)[0] <= 3.0, "mssim fuse cpu vs cuda",
          "mae=" + std::to_string(cv::mean(fusionDiff)[0]));

    const cv::Rect roi = commonOverlapRoi(cv::Size(100, 80), cv::Point2d(10.0, -5.0));
    check(stats, roi == cv::Rect(0, 5, 90, 75), "common overlap roi (10,-5)",
          "roi=(" + std::to_string(roi.x) + "," + std::to_string(roi.y) + "," +
              std::to_string(roi.width) + "x" + std::to_string(roi.height) + ")");
}

void testDefectDetect(TestStats& stats) {
    cv::Mat l2 = cv::Mat::zeros(96, 128, CV_32F);
    const DefectDetectOutput empty = detectDefects(l2, l2);
    check(stats, empty.ok, "defect empty L2 ok", empty.error);
    check(stats, empty.candidates.empty(), "defect empty L2 has no candidates",
          "n=" + std::to_string(empty.candidates.size()));

    cv::rectangle(l2, cv::Rect(40, 30, 18, 14), cv::Scalar(40.0), cv::FILLED);
    const DefectDetectOutput blob = detectDefects(l2, l2);
    check(stats, blob.ok && blob.candidates.size() == 1, "defect recovers one blob",
          "n=" + std::to_string(blob.candidates.size()));
    if (!blob.candidates.empty()) {
        check(stats, blob.candidates[0].area >= 8, "defect blob area",
              "area=" + std::to_string(blob.candidates[0].area));
        check(stats, blob.adaptiveMask.size() == l2.size() && blob.morphMask.size() == l2.size() &&
                         blob.blobMask.size() == l2.size(),
              "defect mask sizes", "mask size mismatch");
    }

    cv::Mat speck = cv::Mat::zeros(64, 64, CV_32F);
    speck.at<float>(20, 20) = 80.0f;
    speck.at<float>(40, 41) = 80.0f;
    const cv::Mat filtered = blobFilter(morphologicalClean(adaptiveThresholdDiff(speck)));
    check(stats, cv::countNonZero(filtered) == 0, "defect blob filter drops speckles",
          "nonzero=" + std::to_string(cv::countNonZero(filtered)));

    cv::Mat large = cv::Mat::zeros(160, 200, CV_32F);
    cv::rectangle(large, cv::Rect(20, 18, 90, 88), cv::Scalar(40.0), cv::FILLED);
    const DefectDetectOutput huge = detectDefects(large, large);
    check(stats, huge.ok && huge.candidates.size() == 1, "defect recovers large filled square",
          "n=" + std::to_string(huge.candidates.size()));
    if (!huge.candidates.empty()) {
        check(stats, huge.candidates[0].area >= 5000, "defect large square area",
              "area=" + std::to_string(huge.candidates[0].area));
    }

    cv::Mat tmpl(96, 128, CV_8UC1, cv::Scalar(120));
    cv::Mat data = tmpl.clone();
    cv::rectangle(data, cv::Rect(20, 20, 18, 16), cv::Scalar(220), cv::FILLED);
    cv::rectangle(data, cv::Rect(80, 50, 18, 16), cv::Scalar(20), cv::FILLED);
    cv::Mat absDiff;
    cv::Mat polarL2;
    cv::absdiff(tmpl, data, absDiff);
    absDiff.convertTo(polarL2, CV_32F);
    DefectDetectParams polar;
    polar.signedDiff.T = 8.0;
    polar.blob.minArea = 20;
    polar.blob.maxArea = 0;
    polar.blob.minWidth = 4;
    polar.blob.minHeight = 4;
    const DefectDetectOutput both = detectDefects(polarL2, tmpl, polar, tmpl, data);
    check(stats, both.ok && both.candidates.size() == 2, "defect bright+dark count",
          "n=" + std::to_string(both.candidates.size()) + " T=" + std::to_string(both.polarityT));
    bool sawBright = false;
    bool sawDark = false;
    for (const auto& c : both.candidates) {
        sawBright = sawBright || c.polarity == DefectPolarity::Bright;
        sawDark = sawDark || c.polarity == DefectPolarity::Dark;
    }
    check(stats, sawBright, "defect classifies bright", "missing bright");
    check(stats, sawDark, "defect classifies dark", "missing dark");
    check(stats, cv::countNonZero(both.brightMask) > 0 && cv::countNonZero(both.darkMask) > 0,
          "defect polarity masks", "bright/dark mask empty");

    polar.filter.keepBright = true;
    polar.filter.keepDark = false;
    polar.filter.keepMixed = false;
    polar.filter.keepUnknown = false;
    const DefectDetectOutput brightOnly = detectDefects(polarL2, tmpl, polar, tmpl, data);
    check(stats, brightOnly.ok && brightOnly.candidates.size() == 1 &&
                     brightOnly.candidates[0].polarity == DefectPolarity::Bright,
          "defect filter keeps bright only",
          "n=" + std::to_string(brightOnly.candidates.size()));

    cv::Mat darkTmpl(202, 202, CV_8UC1, cv::Scalar(186));
    cv::Mat darkData = darkTmpl.clone();
    cv::circle(darkData, cv::Point(115, 119), 5, cv::Scalar(140), cv::FILLED);
    cv::Mat darkAbs;
    cv::Mat darkL2;
    cv::absdiff(darkTmpl, darkData, darkAbs);
    darkAbs.convertTo(darkL2, CV_32F);
    DefectDetectParams darkP;
    darkP.signedDiff.T = 8.0;
    darkP.signedDiff.darkT = 5.0;
    darkP.darkSpot.enableBlackHat = true;
    darkP.darkSpot.kernel = 15;
    darkP.darkSpot.minResponse = 5.0;
    darkP.morph.openSize = 5;
    darkP.morph.closeSize = 5;
    darkP.morph.darkOpenSize = 3;
    darkP.blob.minArea = 15;
    darkP.blob.maxArea = 0;
    darkP.blob.minWidth = 4;
    darkP.blob.minHeight = 4;
    darkP.blob.maxAspectRatio = 3.0;
    darkP.blob.borderMargin = 3;
    darkP.filter.keepBright = false;
    darkP.filter.keepDark = true;
    darkP.filter.keepMixed = false;
    darkP.filter.keepUnknown = false;
    const DefectDetectOutput smallDark = detectDefects(darkL2, darkTmpl, darkP, darkTmpl, darkData);
    bool foundSmallDark = false;
    for (const auto& c : smallDark.candidates) {
        if (c.polarity == DefectPolarity::Dark && c.bbox.contains(cv::Point(115, 119))) {
            foundSmallDark = true;
        }
    }
    check(stats, smallDark.ok && foundSmallDark, "defect recovers ~10px dark spot",
          "n=" + std::to_string(smallDark.candidates.size()) +
              " polarityT=" + std::to_string(smallDark.polarityT));

    cv::Mat strip = cv::Mat::zeros(120, 220, CV_32F);
    cv::rectangle(strip, cv::Rect(20, 40, 180, 7), cv::Scalar(40.0), cv::FILLED);
    DefectDetectParams stripP;
    stripP.blob.minArea = 20;
    stripP.blob.maxArea = 0;
    stripP.blob.minWidth = 4;
    stripP.blob.minHeight = 4;
    stripP.blob.maxAspectRatio = 3.0;
    stripP.blob.borderMargin = 0;
    const DefectDetectOutput stripOut = detectDefects(strip, strip, stripP);
    check(stats, stripOut.ok && stripOut.candidates.empty(),
          "defect drops large high-aspect strip",
          "n=" + std::to_string(stripOut.candidates.size()));

    cv::Mat edge = cv::Mat::zeros(80, 80, CV_32F);
    cv::rectangle(edge, cv::Rect(0, 10, 6, 50), cv::Scalar(40.0), cv::FILLED);
    DefectDetectParams edgeP;
    edgeP.blob.minArea = 10;
    edgeP.blob.maxArea = 0;
    edgeP.blob.minWidth = 4;
    edgeP.blob.minHeight = 4;
    edgeP.blob.maxAspectRatio = 12.0;
    edgeP.blob.borderMargin = 2;
    edgeP.blob.borderStripAspect = 2.5;
    const DefectDetectOutput edgeOut = detectDefects(edge, edge, edgeP);
    check(stats, edgeOut.ok && edgeOut.candidates.empty(), "defect drops border-touching strip",
          "n=" + std::to_string(edgeOut.candidates.size()));

    cv::Mat edgeDot = cv::Mat::zeros(96, 128, CV_32F);
    cv::rectangle(edgeDot, cv::Rect(21, 0, 16, 10), cv::Scalar(30.0), cv::FILLED);
    DefectDetectParams edgeDotP;
    edgeDotP.morph.openSize = 3;
    edgeDotP.morph.closeSize = 3;
    edgeDotP.blob.minArea = 20;
    edgeDotP.blob.maxArea = 0;
    edgeDotP.blob.minWidth = 4;
    edgeDotP.blob.minHeight = 4;
    edgeDotP.blob.maxAspectRatio = 3.0;
    edgeDotP.blob.borderMargin = 4;
    edgeDotP.blob.borderStripAspect = 2.5;
    const DefectDetectOutput edgeDotOut = detectDefects(edgeDot, edgeDot, edgeDotP);
    bool foundEdgeDot = false;
    for (const auto& c : edgeDotOut.candidates) {
        if (c.bbox.contains(cv::Point(28, 4))) {
            foundEdgeDot = true;
        }
    }
    check(stats, edgeDotOut.ok && foundEdgeDot, "defect keeps compact dark spot on die edge",
          "n=" + std::to_string(edgeDotOut.candidates.size()));

    cv::Mat whiteDie(200, 200, CV_8UC1, cv::Scalar(80));
    cv::rectangle(whiteDie, cv::Rect(10, 20, 120, 140), cv::Scalar(245), cv::FILLED);
    LargeWhiteParams lw;
    lw.enable = true;
    lw.threshold = 230.0;
    lw.minArea = 4000;
    lw.minFill = 0.15;
    lw.maxAspectRatio = 3.0;
    const std::vector<DefectCandidate> whites = extractLargeWhiteRegions(whiteDie, lw);
    check(stats, whites.size() == 1 && whites[0].polarity == DefectPolarity::Bright &&
                     whites[0].area >= 4000,
          "large white recovers filled bright region",
          "n=" + std::to_string(whites.size()) +
              " area=" + (whites.empty() ? "0" : std::to_string(whites[0].area)));

    cv::Mat whiteStrip(200, 200, CV_8UC1, cv::Scalar(80));
    cv::rectangle(whiteStrip, cv::Rect(5, 80, 190, 30), cv::Scalar(245), cv::FILLED);
    const std::vector<DefectCandidate> strips = extractLargeWhiteRegions(whiteStrip, lw);
    check(stats, strips.empty(), "large white drops high-aspect bright strip",
          "n=" + std::to_string(strips.size()));

    LargeWhiteParams lwOff = lw;
    lwOff.enable = false;
    check(stats, extractLargeWhiteRegions(whiteDie, lwOff).empty(),
          "large white skipped when disabled", "still emitted candidates");
}

void testCandidateLabelReuse(TestStats& stats) {
    cv::Mat mask = cv::Mat::zeros(128, 160, CV_8UC1);
    cv::rectangle(mask, cv::Rect(12, 15, 20, 18), cv::Scalar(255), cv::FILLED);
    cv::rectangle(mask, cv::Rect(91, 70, 31, 24), cv::Scalar(255), cv::FILLED);
    cv::Mat l2(mask.size(), CV_32FC1, cv::Scalar(25.0f));
    cv::Mat labels;
    const std::vector<DefectCandidate> candidates =
        extractDefectCandidates(mask, l2, {}, {}, {}, &labels);
    const cv::Mat rebuilt = maskFromCandidates(mask.size(), candidates, labels);
    check(stats, candidates.size() == 2, "candidate extraction component count",
          "n=" + std::to_string(candidates.size()));
    check(stats, !labels.empty() && labels.type() == CV_32SC1,
          "candidate extraction returns labels", "labels missing or wrong type");
    check(stats, cv::countNonZero(mask != rebuilt) == 0, "candidate label reuse exact mask",
          "rebuilt mask differs");

    cv::Mat wrongSize(32, 32, CV_32FC1, cv::Scalar(1.0f));
    labels = cv::Mat::ones(mask.size(), CV_32SC1);
    const auto invalid = extractDefectCandidates(mask, wrongSize, {}, {}, {}, &labels);
    check(stats, invalid.empty() && labels.empty(), "candidate rejects mismatched maps",
          "invalid input was accepted");
}

cv::Mat makeScene(int width, int height, bool color) {
    cv::Mat gray = makePattern(width, height, false);
    if (!color) {
        return gray;
    }
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    for (int y = height * 3 / 5; y < height * 4 / 5; ++y) {
        for (int x = width * 3 / 5; x < width * 5 / 6; ++x) {
            cv::Vec3b& p = bgr.at<cv::Vec3b>(y, x);
            p[0] = static_cast<unsigned char>(std::min(255, p[0] + 35));
            p[2] = static_cast<unsigned char>(std::max(0, p[2] - 20));
        }
    }
    return bgr;
}

double overlapImageMae(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.type() != b.type() || a.channels() != b.channels()) {
        return 1e9;
    }
    const int w = std::min(a.cols, b.cols);
    const int h = std::min(a.rows, b.rows);
    if (w < 8 || h < 8) {
        return 1e9;
    }
    cv::Mat diff;
    cv::absdiff(a(cv::Rect(0, 0, w, h)), b(cv::Rect(0, 0, w, h)), diff);
    return cv::mean(diff)[0];
}

void testEndToEndPipeline(TestStats& stats, PipelineContext& ctx) {
    printf("\n--- end-to-end iterative (cpu vs cuda) ---\n");

    struct Case {
        const char* name;
        int size;
        cv::Point2d gt;
        bool color;
        PipelineAlign align;
        int repeats;
    };

    const Case cases[] = {
        {"gray/phase/(3,-2)", 256, {3.0, -2.0}, false, PipelineAlign::Phase, 2},
        {"gray/phase/(2,3)", 256, {2.0, 3.0}, false, PipelineAlign::Phase, 2},
        {"gray/phase/(-4,1)", 256, {-4.0, 1.0}, false, PipelineAlign::Phase, 2},
        {"gray/ecc/(3,-2)", 256, {3.0, -2.0}, false, PipelineAlign::Ecc, 2},
        {"gray/ecc/(1.5,2.5)", 256, {1.5, 2.5}, false, PipelineAlign::Ecc, 2},
        {"color/phase/(2,-3)", 256, {2.0, -3.0}, true, PipelineAlign::Phase, 2},
        {"color/ecc/(2.5,1.5)", 256, {2.5, 1.5}, true, PipelineAlign::Ecc, 2},
        {"gray/phase/320", 320, {4.0, 3.0}, false, PipelineAlign::Phase, 2},
    };

    printf("%-22s %6s %5s %16s %8s %8s\n", "case", "be", "T", "shift", "cands", "status");
    printf("%-22s %6s %5s %16s %8s %8s\n", "----------------------", "------", "-----",
           "----------------", "--------", "--------");

    for (const Case& c : cases) {
        const cv::Mat tmpl = makeScene(c.size, c.size, c.color);
        const cv::Mat data = shifted(tmpl, c.gt, cv::INTER_LINEAR);
        PipelineOutput prevCpu;
        PipelineOutput prevGpu;
        const bool phase = c.align == PipelineAlign::Phase;
        const double shiftTol = phase ? 0.45 : 0.35;
        const double agreeTol = phase ? 0.30 : 0.20;

        for (int r = 0; r < c.repeats; ++r) {
            char tag[96];
            std::snprintf(tag, sizeof(tag), "%s#%d", c.name, r + 1);

            const PipelineOutput cpu =
                runD2DPipeline(tmpl, data, PipelineBackend::Cpu, c.align, ctx);
            const PipelineOutput gpu =
                runD2DPipeline(tmpl, data, PipelineBackend::Cuda, c.align, ctx);

            check(stats, cpu.ok, std::string("e2e cpu ") + tag, cpu.error);
            check(stats, gpu.ok, std::string("e2e cuda ") + tag, gpu.error);
            if (!cpu.ok || !gpu.ok) {
                continue;
            }

            check(stats, cpu.templateT == gpu.templateT && cpu.dataT == gpu.dataT,
                  std::string("e2e T cpu vs cuda ") + tag,
                  "cpu=(" + std::to_string(cpu.templateT) + "," + std::to_string(cpu.dataT) +
                      ") cuda=(" + std::to_string(gpu.templateT) + "," +
                      std::to_string(gpu.dataT) + ")");

            if (phase) {
                checkEquivalentShift(stats, cpu.shift, c.gt, shiftTol,
                                     std::string("e2e cpu recover ") + tag);
                checkEquivalentShift(stats, gpu.shift, c.gt, shiftTol,
                                     std::string("e2e cuda recover ") + tag);
                checkEquivalentShift(stats, gpu.shift, cpu.shift, agreeTol,
                                     std::string("e2e shift cpu vs cuda ") + tag);
            } else {
                checkEquivalentShift(stats, cpu.shift, c.gt, shiftTol,
                                     std::string("e2e cpu recover ") + tag);
                checkEquivalentShift(stats, gpu.shift, c.gt, shiftTol,
                                     std::string("e2e cuda recover ") + tag);
                checkShift(stats, gpu.shift, cpu.shift, agreeTol,
                           std::string("e2e shift cpu vs cuda ") + tag);
            }

            check(stats, cpu.adaptiveMask.size() == cpu.templateCommon.size() &&
                             gpu.adaptiveMask.size() == gpu.templateCommon.size(),
                  std::string("e2e adaptive size ") + tag, "mask size mismatch");
            check(stats, cpu.morphMask.size() == cpu.templateCommon.size() &&
                             cpu.blobMask.size() == cpu.templateCommon.size(),
                  std::string("e2e morph/blob size ") + tag, "mask size mismatch");
            check(stats,
                  std::abs(static_cast<int>(cpu.candidates.size()) -
                           static_cast<int>(gpu.candidates.size())) <= 3,
                  std::string("e2e candidates cpu vs cuda ") + tag,
                  "cpu=" + std::to_string(cpu.candidates.size()) +
                      " cuda=" + std::to_string(gpu.candidates.size()));

            const double same = std::hypot(gpu.shift.x - cpu.shift.x, gpu.shift.y - cpu.shift.y);
            if (same <= agreeTol && cpu.commonRoi.x == gpu.commonRoi.x &&
                cpu.commonRoi.y == gpu.commonRoi.y && cpu.blobMask.size() == gpu.blobMask.size()) {
                const double maskErr = overlapImageMae(cpu.blobMask, gpu.blobMask);
                check(stats, maskErr <= 8.0, std::string("e2e blob mask cpu vs cuda ") + tag,
                      "mae=" + std::to_string(maskErr));
            }

            if (r > 0 && prevCpu.ok) {
                checkShift(stats, cpu.shift, prevCpu.shift, 1e-6,
                           std::string("e2e cpu stable ") + tag);
                check(stats, cpu.candidates.size() == prevCpu.candidates.size(),
                      std::string("e2e cpu candidates stable ") + tag,
                      "n=" + std::to_string(cpu.candidates.size()));
            }
            if (r > 0 && prevGpu.ok) {
                checkShift(stats, gpu.shift, prevGpu.shift, 1e-6,
                           std::string("e2e cuda stable ") + tag);
                check(stats, gpu.candidates.size() == prevGpu.candidates.size(),
                      std::string("e2e cuda candidates stable ") + tag,
                      "n=" + std::to_string(gpu.candidates.size()));
            }

            if (r == 0) {
                printf("%-22s %6s %5d (%6.2f,%6.2f) %8zu %8s\n", c.name, "cpu", cpu.templateT,
                       cpu.shift.x, cpu.shift.y, cpu.candidates.size(), cpu.ok ? "ok" : "fail");
                printf("%-22s %6s %5d (%6.2f,%6.2f) %8zu %8s\n", c.name, "cuda", gpu.templateT,
                       gpu.shift.x, gpu.shift.y, gpu.candidates.size(), gpu.ok ? "ok" : "fail");
            }
            prevCpu = cpu;
            prevGpu = gpu;
        }
    }
}

void testPipelineRejectsFlatPhaseInput(TestStats& stats, PipelineContext& ctx) {
    const cv::Mat flat(128, 128, CV_8UC1, cv::Scalar(80));
    const PipelineOutput cpu =
        runD2DPipeline(flat, flat, PipelineBackend::Cpu, PipelineAlign::Phase, ctx);
    const PipelineOutput gpu =
        runD2DPipeline(flat, flat, PipelineBackend::Cuda, PipelineAlign::Phase, ctx);
    check(stats, !cpu.ok && !gpu.ok, "phase pipeline rejects flat input",
          "cpu=" + std::to_string(cpu.ok) + " cuda=" + std::to_string(gpu.ok));
}

void testInspectWaferOverlay(TestStats& stats) {
    printf("\n--- inspect wafer overlay (merged c3000 path) ---\n");

    const cv::Mat pattern = makePattern(160, 160, false);
    cv::Mat full(160, 320, CV_8UC1);
    pattern.copyTo(full(cv::Rect(0, 0, 160, 160)));
    cv::Mat dataDie = pattern.clone();
    const cv::Rect localDefect(52, 44, 22, 18);
    cv::rectangle(dataDie, localDefect, cv::Scalar(8), cv::FILLED);
    dataDie.copyTo(full(cv::Rect(160, 0, 160, 160)));

    cutdie::CutResult cut;
    cut.full = full;
    cutdie::DieBox left;
    left.x_idx = 0;
    left.y_idx = 0;
    left.right = 160;
    left.bottom = 160;
    cutdie::DieBox right;
    right.x_idx = 1;
    right.y_idx = 0;
    right.left = 160;
    right.right = 320;
    right.bottom = 160;
    cut.boxes = {left, right};

    cutdie::Options opt;
    opt.out = std::filesystem::temp_directory_path() / "c3000_inspect_selfcheck";
    opt.keep_skipped = false;
    opt.align_method = cutdie::AlignMethod::EccCpu;
    opt.histogram_method = cutdie::HistogramMethod::Cpu;
    opt.normalize_method = cutdie::NormalizeMethod::Cpu;
    opt.defect.signedDiff.T = 8.0;
    opt.defect.blob.minArea = 20;
    opt.defect.blob.maxArea = 0;

    cutdie::TemplateCache templates;
    templates.single = true;
    templates.source = "synthetic-pattern";
    templates.single_img = pattern;

    std::error_code ec;
    std::filesystem::create_directories(opt.out, ec);
    const cutdie::InspectResult result = cutdie::inspect_wafer(opt, cut, templates);

    check(stats, result.failed == 0, "inspect wafer no die failures",
          "failed=" + std::to_string(result.failed));
    check(stats, result.inspected == 2, "inspect wafer processes both dies",
          "inspected=" + std::to_string(result.inspected));
    check(stats, std::filesystem::exists(result.overlay_path), "inspect wafer overlay saved",
          result.overlay_path.string());
    check(stats, std::filesystem::exists(result.csv_path), "inspect wafer csv saved",
          result.csv_path.string());

    int onRight = 0;
    bool mappedToOriginal = false;
    bool sawDark = false;
    for (const auto& d : result.defects) {
        if (d.die_x == 1) {
            ++onRight;
            sawDark = sawDark || d.polarity == DefectPolarity::Dark;
            if (d.bbox.x >= 160 && d.bbox.x + d.bbox.width <= 320 && d.bbox.y >= 0 &&
                d.bbox.y + d.bbox.height <= 160) {
                mappedToOriginal = true;
            }
        }
    }
    check(stats, onRight >= 1, "inspect wafer finds defect on data die",
          "right_die_defects=" + std::to_string(onRight) +
              " total=" + std::to_string(result.defect_count));
    check(stats, mappedToOriginal, "inspect wafer maps defect onto original image",
          "no right-die bbox inside original crop");
    check(stats, sawDark, "inspect wafer classifies dark defect", "no dark polarity on data die");
    check(stats, !result.die_overlay_dir.empty() && std::filesystem::exists(result.die_overlay_dir),
          "inspect wafer writes per-die overlays", result.die_overlay_dir.string());
}

void testRealSmallDarkSpot(TestStats& stats, PipelineContext& ctx) {
    const std::string fixtureDir = environmentValue("C3000_TEST_DATA_DIR");
    if (fixtureDir.empty()) {
        printf("skip real dark-spot (C3000_TEST_DATA_DIR not set)\n");
        return;
    }
    const std::filesystem::path fixtureRoot(fixtureDir);
    const std::filesystem::path dataPath = fixtureRoot / "0050_die_x12_y2.png";
    const std::filesystem::path tmplPath = fixtureRoot / "0050_die_x0_y12.png";
    if (!std::filesystem::exists(dataPath) || !std::filesystem::exists(tmplPath)) {
        printf("skip real dark-spot (fixture files missing in C3000_TEST_DATA_DIR)\n");
        return;
    }

    const cv::Mat tmpl = cv::imread(tmplPath.string(), cv::IMREAD_UNCHANGED);
    const cv::Mat data = cv::imread(dataPath.string(), cv::IMREAD_UNCHANGED);
    check(stats, !tmpl.empty() && !data.empty(), "real dark-spot images load",
          "tmpl=" + tmplPath.string() + " data=" + dataPath.string());
    if (tmpl.empty() || data.empty()) {
        return;
    }

    DefectDetectParams p;
    p.signedDiff.T = 8.0;
    p.signedDiff.darkT = 5.0;
    p.darkSpot.enableBlackHat = true;
    p.darkSpot.kernel = 15;
    p.darkSpot.minResponse = 5.0;
    p.morph.openSize = 3;
    p.morph.closeSize = 5;
    p.morph.darkOpenSize = 3;
    p.blob.minArea = 15;
    p.blob.maxArea = 0;
    p.blob.minWidth = 4;
    p.blob.minHeight = 4;
    p.blob.maxAspectRatio = 3.0;
    p.blob.borderMargin = 3;
    p.filter.keepBright = false;
    p.filter.keepDark = true;
    p.filter.keepMixed = false;
    p.filter.keepUnknown = false;
    ctx.defectParams = p;

    const PipelineOutput pipe =
        runD2DPipeline(tmpl, data, PipelineBackend::Cpu, PipelineAlign::Phase, ctx);
    check(stats, pipe.ok, "real dark-spot pipeline ok", pipe.error);

    bool found = false;
    const cv::Point expect(115, 119);
    for (const auto& c : pipe.candidates) {
        const cv::Point2d delta = c.centroid - cv::Point2d(expect);
        if (c.polarity == DefectPolarity::Dark &&
            (c.bbox.contains(expect) || std::sqrt(delta.x * delta.x + delta.y * delta.y) <= 20.0)) {
            found = true;
            break;
        }
    }
    check(stats, found, "real 0050_die_x12_y2 small dark spot",
          "n=" + std::to_string(pipe.candidates.size()));
}

}  // namespace

int runRegressionTests() {
    printf("\n=== regression tests ===\n");
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed\n");
        return 1;
    }

    TestStats stats;
    PhaseCorrelateCuda phaseCuda;
    EccAlignCuda eccCuda;
    IntensityNormalizeCuda intensityCuda;
    MssimCuda mssimCuda;
    HistogramThreshold hist;

    PipelineContext ctx;
    ctx.hist = &hist;
    ctx.phaseCuda = &phaseCuda;
    ctx.eccCuda = &eccCuda;
    ctx.intensityCuda = &intensityCuda;

    testCommandLine(stats);
    testPhaseZeroShift(stats, phaseCuda);
    testPhaseKnownShifts(stats, phaseCuda);
    testPhaseSubpixel(stats, phaseCuda);
    testPhaseLargeShifts(stats, phaseCuda);
    testEccShift(stats, eccCuda);
    testEccWorkspaceIsolation(stats);
    testHistogramRules(stats, hist);
    testHistogramApplyAndColor(stats, hist);
    testIntensityNormalize(stats, intensityCuda);
    testMssimAndOverlap(stats, mssimCuda);
    testDefectDetect(stats);
    testCandidateLabelReuse(stats);
    testPipelineRejectsFlatPhaseInput(stats, ctx);
    testEndToEndPipeline(stats, ctx);
    testInspectWaferOverlay(stats);
    testRealSmallDarkSpot(stats, ctx);

    printf("\nregression: %d passed, %d failed\n", stats.passed, stats.failed);
    return stats.failed == 0 ? 0 : 1;
}

int runBenchmark(int iters) {
    if (iters <= 0) {
        fprintf(stderr, "benchmark iterations must be positive\n");
        return 2;
    }
    printf("\n=== benchmark (median of %d iters, after 1 warmup) ===\n", iters);
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed\n");
        return 1;
    }
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) == cudaSuccess) {
        printf("device: %s, compute %d.%d, %.0f MiB global memory\n", device.name,
               device.major, device.minor,
               static_cast<double>(device.totalGlobalMem) / (1024.0 * 1024.0));
    }

    PhaseCorrelateCuda phaseCuda;
    EccAlignCuda eccCuda;
    IntensityNormalizeCuda intensityCuda;
    MssimCuda mssimCuda;
    HistogramThreshold hist;
    const int sizes[] = {256, 512, 1024};

    printf("%-8s %12s %12s %10s %12s %12s\n", "size", "phase-cpu", "phase-cuda", "speedup", "hist",
           "apply");
    printf("%-8s %12s %12s %10s %12s %12s\n", "--------", "------------", "------------",
           "----------", "------------", "------------");

    for (int size : sizes) {
        const cv::Mat tmplBin = makePattern(size, size, true);
        const cv::Mat dataBin = shifted(tmplBin, cv::Point2d(4.0, -3.0), cv::INTER_NEAREST);
        const cv::Mat tmplGray = makePattern(size, size, false);
        const int nPixels = size * size;
        HistogramThresholdResult histResult{};
        PhaseCorrelateResult phaseResult{};
        cudaError_t phaseStatus = cudaSuccess;
        cudaError_t histogramStatus = cudaSuccess;
        cudaError_t applyStatus = cudaSuccess;
        cv::Mat binary(size, size, CV_8UC1);

        const double cpuMs = benchmarkMedian(iters, false, [&]() {
            phaseShiftBinary(tmplBin, dataBin, nullptr);
        });
        const double gpuMs = benchmarkMedian(iters, true, [&]() {
            phaseStatus = phaseCuda.compute(tmplBin.data, dataBin.data, size, size, phaseResult);
        });
        const double histMs = benchmarkMedian(iters, true,
            [&]() { histogramStatus = hist.computeFromGray(tmplGray.data, size, size, histResult); });
        const double applyMs = benchmarkMedian(iters, true, [&]() {
            applyStatus = hist.apply(tmplGray.data, nPixels,
                                     histResult.threshold > 0 ? histResult.threshold : 125,
                                     binary.data);
        });
        if (phaseStatus != cudaSuccess || histogramStatus != cudaSuccess ||
            applyStatus != cudaSuccess) {
            fprintf(stderr, "phase/histogram benchmark failed at %d^2\n", size);
            return 1;
        }

        const double speedup = gpuMs > 1e-9 ? cpuMs / gpuMs : 0.0;
        printf("%4d^2 %10.3f ms %10.3f ms %9.2fx %10.3f ms %10.3f ms\n", size, cpuMs, gpuMs, speedup,
               histMs, applyMs);
    }

    printf("\n=== connected-components candidate extraction ===\n");
    printf("%-8s %14s %14s %10s\n", "size", "legacy-2pass", "reuse-1pass", "speedup");
    printf("%-8s %14s %14s %10s\n", "--------", "--------------", "--------------",
           "----------");
    for (int size : sizes) {
        cv::Mat mask = cv::Mat::zeros(size, size, CV_8UC1);
        for (int y = 8; y + 8 < size; y += 24) {
            for (int x = 8; x + 8 < size; x += 24) {
                cv::rectangle(mask, cv::Rect(x, y, 7, 7), cv::Scalar(255), cv::FILLED);
            }
        }
        cv::Mat l2(mask.size(), CV_32FC1, cv::Scalar(25.0f));
        BlobFilterParams filter;
        filter.minArea = 1;
        filter.maxArea = 0;
        filter.minWidth = 1;
        filter.minHeight = 1;
        filter.maxAspectRatio = 100.0;
        filter.borderMargin = 0;
        cv::Mat legacyMask;
        cv::Mat reuseMask;

        const double legacyMs = benchmarkMedian(iters, false, [&]() {
            cv::Mat labels;
            cv::Mat componentStats;
            cv::Mat centroids;
            cv::connectedComponentsWithStats(mask, labels, componentStats, centroids, 8, CV_32S);
            const auto raw = extractDefectCandidates(mask, l2);
            const auto kept = filterDefects(raw, filter, {}, mask.size());
            legacyMask = maskFromCandidates(mask.size(), kept, labels);
        });
        const double reuseMs = benchmarkMedian(iters, false, [&]() {
            cv::Mat labels;
            const auto raw = extractDefectCandidates(mask, l2, {}, {}, {}, &labels);
            const auto kept = filterDefects(raw, filter, {}, mask.size());
            reuseMask = maskFromCandidates(mask.size(), kept, labels);
        });
        if (legacyMask.empty() || reuseMask.empty() ||
            cv::countNonZero(legacyMask != reuseMask) != 0) {
            fprintf(stderr, "connected-components benchmark produced mismatched output\n");
            return 1;
        }
        printf("%4d^2 %12.3f ms %12.3f ms %9.2fx\n", size, legacyMs, reuseMs,
               reuseMs > 1e-9 ? legacyMs / reuseMs : 0.0);
    }

    printf("\n%-8s %12s %12s %10s\n", "size", "ecc-cpu", "ecc-cuda", "speedup");
    printf("%-8s %12s %12s %10s\n", "--------", "------------", "------------", "----------");
    for (int size : sizes) {
        if (size > 512) {
            continue;
        }
        const cv::Mat tmplGray = makePattern(size, size, false);
        const cv::Mat dataGray = shifted(tmplGray, cv::Point2d(3.0, -2.0), cv::INTER_LINEAR);
        EccAlignResult cpuResult;
        EccCudaResult gpuResult{};
        cudaError_t gpuStatus = cudaSuccess;
        const double cpuMs =
            benchmarkMedian(iters, false, [&]() { cpuResult = eccShift(tmplGray, dataGray); });
        const double gpuMs = benchmarkMedian(iters, true, [&]() {
            gpuStatus = eccCuda.compute(tmplGray.data, dataGray.data, size, size, gpuResult);
        });
        if (!cpuResult.ok || gpuStatus != cudaSuccess || !gpuResult.ok) {
            fprintf(stderr, "ECC benchmark failed at %d^2\n", size);
            return 1;
        }
        const double speedup = gpuMs > 1e-9 ? cpuMs / gpuMs : 0.0;
        printf("%4d^2 %10.3f ms %10.3f ms %9.2fx\n", size, cpuMs, gpuMs, speedup);
    }

    printf("\n%-8s %8s %14s %14s %10s\n", "size", "channels", "normalize-cpu",
           "normalize-cuda", "speedup");
    printf("%-8s %8s %14s %14s %10s\n", "--------", "--------", "--------------",
           "--------------", "----------");
    for (int size : sizes) {
        const cv::Mat gray = makePattern(size, size, false);
        for (int channels : {1, 3}) {
            cv::Mat reference;
            if (channels == 1) {
                reference = gray;
            } else {
                cv::cvtColor(gray, reference, cv::COLOR_GRAY2BGR);
            }
            cv::Mat source;
            reference.convertTo(source, CV_8U, 0.82, 18.0);
            IntensityNormalizeOutput cpuResult;
            IntensityNormalizeOutput gpuResult;

            const double cpuMs = benchmarkMedian(
                iters, false, [&]() { cpuResult = intensityNormalize(reference, source); });
            const double gpuMs = benchmarkMedian(
                iters, true,
                [&]() { gpuResult = intensityNormalizeCuda(intensityCuda, reference, source); });
            if (!cpuResult.ok || !gpuResult.ok) {
                fprintf(stderr, "normalization benchmark failed at %d^2/%dch\n", size, channels);
                return 1;
            }
            const double speedup = gpuMs > 1e-9 ? cpuMs / gpuMs : 0.0;
            printf("%4d^2 %8d %12.3f ms %12.3f ms %9.2fx\n", size, channels, cpuMs,
                   gpuMs, speedup);
        }
    }

    printf("\n%-8s %12s %12s %10s %12s %12s\n", "size", "mssim-cpu", "mssim-cuda", "speedup",
           "fuse-cpu", "fuse-cuda");
    printf("%-8s %12s %12s %10s %12s %12s\n", "--------", "------------", "------------",
           "----------", "------------", "------------");
    for (int size : sizes) {
        const cv::Mat a = makePattern(size, size, false);
        cv::Mat b = shifted(a, cv::Point2d(2.0, -1.0), cv::INTER_LINEAR);
        MssimResult cpuResult;
        MssimResult gpuResult;
        gpuResult.map.create(size, size, CV_32F);
        cv::Mat fusionCpu;
        cv::Mat fusionGpu(a.size(), a.type());
        cudaError_t computeStatus = cudaSuccess;
        cudaError_t fuseStatus = cudaSuccess;

        const double cpuMs =
            benchmarkMedian(iters, false, [&]() { cpuResult = computeMssim(a, b); });
        const double gpuMs = benchmarkMedian(iters, true, [&]() {
            computeStatus = mssimCuda.compute(a.data, b.data, size, size, 1,
                                              gpuResult.map.ptr<float>(), gpuResult.mssim);
        });
        const double fuseCpuMs = benchmarkMedian(
            iters, false, [&]() { fusionCpu = fuseByMssim(a, b, cpuResult.map); });
        const double fuseGpuMs = benchmarkMedian(iters, true, [&]() {
            fuseStatus = mssimCuda.fuse(a.data, b.data, gpuResult.map.ptr<float>(), size, size, 1,
                                        fusionGpu.data);
        });
        if (cpuResult.map.empty() || computeStatus != cudaSuccess || fuseStatus != cudaSuccess) {
            fprintf(stderr, "MSSIM benchmark failed at %d^2\n", size);
            return 1;
        }
        const double speedup = gpuMs > 1e-9 ? cpuMs / gpuMs : 0.0;
        printf("%4d^2 %10.3f ms %10.3f ms %9.2fx %10.3f ms %10.3f ms\n", size, cpuMs, gpuMs, speedup,
               fuseCpuMs, fuseGpuMs);
    }

    printf("\n=== end-to-end pipeline benchmark (median of %d iters, after 1 warmup) ===\n", iters);
    PipelineContext ctx;
    ctx.hist = &hist;
    ctx.phaseCuda = &phaseCuda;
    ctx.eccCuda = &eccCuda;
    ctx.intensityCuda = &intensityCuda;

    printf("%-8s %-6s %-7s %10s %9s %9s %9s %9s %9s\n", "size", "align", "backend",
           "total", "hist", "align", "norm+diff", "adapt", "blob");
    printf("%-8s %-6s %-7s %10s %9s %9s %9s %9s %9s\n", "--------", "------",
           "-------", "----------", "---------", "---------", "---------", "---------",
           "---------");

    const PipelineAlign aligns[] = {PipelineAlign::Phase, PipelineAlign::Ecc};
    for (PipelineAlign align : aligns) {
        for (int size : sizes) {
            if (align == PipelineAlign::Ecc && size > 512) {
                continue;
            }
            const cv::Mat tmpl = makeScene(size, size, false);
            const cv::Mat data = shifted(tmpl, cv::Point2d(3.0, -2.0), cv::INTER_LINEAR);
            PipelineOutput cpuOut;
            PipelineOutput gpuOut;

            const double cpuMs = benchmarkMedian(iters, false, [&]() {
                cpuOut = runD2DPipeline(tmpl, data, PipelineBackend::Cpu, align, ctx, false);
            });
            const double gpuMs = benchmarkMedian(iters, true, [&]() {
                gpuOut = runD2DPipeline(tmpl, data, PipelineBackend::Cuda, align, ctx, false);
            });
            if (!cpuOut.ok || !gpuOut.ok) {
                fprintf(stderr, "pipeline benchmark failed at %d^2 (%s): cpu=%s cuda=%s\n", size,
                        pipelineAlignName(align), cpuOut.error.c_str(), gpuOut.error.c_str());
                return 1;
            }
            // Stage timings are sampled separately because enabling them synchronizes CUDA at
            // every stage and would otherwise distort the end-to-end median above.
            cpuOut = runD2DPipeline(tmpl, data, PipelineBackend::Cpu, align, ctx, true);
            gpuOut = runD2DPipeline(tmpl, data, PipelineBackend::Cuda, align, ctx, true);
            if (!cpuOut.ok || !gpuOut.ok) {
                fprintf(stderr, "pipeline stage timing failed at %d^2 (%s)\n", size,
                        pipelineAlignName(align));
                return 1;
            }
            const double speedup = gpuMs > 1e-9 ? cpuMs / gpuMs : 0.0;
            printf(" %4d^2 %-6s %-7s %8.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n", size,
                   pipelineAlignName(align), "cpu", cpuMs, cpuOut.timing.histMs,
                   cpuOut.timing.alignMs, cpuOut.timing.normalizeDiffMs,
                   cpuOut.timing.adaptiveMs, cpuOut.timing.blobMs);
            printf(" %4d^2 %-6s %-7s %8.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n", size,
                   pipelineAlignName(align), "cuda", gpuMs, gpuOut.timing.histMs,
                   gpuOut.timing.alignMs, gpuOut.timing.normalizeDiffMs,
                   gpuOut.timing.adaptiveMs, gpuOut.timing.blobMs);
            printf("        %-6s %-7s %8.2fx\n", "", "speedup", speedup);
        }
    }
    return 0;
}
