#pragma once

#include "DefectDetect.hpp"
#include "EccAlign.hpp"
#include "HistogramThreshold.h"
#include "BrightDarkDetector.h"
#include "IntensityNormalizer.h"

#include <filesystem>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cutdie {

namespace fs = std::filesystem;

inline constexpr int kDetectTarget = 8000;
inline constexpr const char* kDefaultDir = "C:/Users/aspiring135/Desktop/datasets/10um-gray";

enum class AlignMethod { PhaseCuda, PhaseCpu, EccCuda, EccCpu };
enum class HistogramMethod { Auto, Cpu, Cuda };
enum class NormalizeMethod { Auto, Cpu, Cuda };
enum class InspectMode { SingleDie,DieToDie, DieToDatabase, YoloOpenVino, AnomalibOpenVino};

struct DieMap {
    double wafer_diameter = 0;
    double edge_offset = 0;
    double pitch_x = 0;
    double pitch_y = 0;
    double origin_offset_x = 0;
    double origin_offset_y = 0;
    double scribe_w = 0;
    double scribe_h = 0;
    double trim_cx = 0;
    double trim_cy = 0;
    double trim_size = 0;
    int gray_min = 0;
    int gray_max = 255;
    bool patterned = true;
    std::set<std::pair<int, int>> skipped;
};

struct WaferCircle {
    double cx = 0;
    double cy = 0;
    double radius = 0;
    int lut = 0;
};

struct DieBox {
    int x_idx = 0;
    int y_idx = 0;
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
    bool skipped = false;

    int width() const { return right - left; }
    int height() const { return bottom - top; }
    std::string file_name(const std::string& prefix = {}) const {
        std::string name = "die_x" + std::to_string(x_idx) + "_y" + std::to_string(y_idx) + ".png";
        if (prefix.empty()) {
            return name;
        }
        return prefix + "_" + name;
    }
};

struct Options {
    fs::path dir = kDefaultDir;
    fs::path image;
    fs::path xml;
    fs::path out;
    bool keep_skipped = false;
    // 这些晶粒不检出。坐标与 patch 名 die_x*_y* 一致。
    std::set<std::pair<int, int>> skip_inspect;
    // 非空时只检出这些晶粒。
    std::set<std::pair<int, int>> inspect_only;
    bool recursive = false;
    double resolution_um = 10.0;  // 每个像素对应多少微米 (um/px)

    // 切割并保存全部 patch。空则写到 <out>/patches
    fs::path patch_out;

    // 外部模板：单张图（所有 die 共用）或文件夹（按 die_x*_y* 匹配）
    fs::path template_path;
    bool inspect = true;
    bool template_neighbor = false;   // 文件夹模板时，用相邻 die 而不是自己
    int template_neighbor_min = 1;    // >=2：至少两个邻居都检出才算有缺陷
    AlignMethod align_method = AlignMethod::EccCpu;
    HistogramMethod histogram_method = HistogramMethod::Cpu;
    NormalizeMethod normalize_method = NormalizeMethod::Cpu;

    // 非空则只评测已切割 patch，不切晶圆。
    // eval_truth_dir：正样本图像目录，或 image_names.txt（每行一个文件名视为有缺陷）。
    fs::path eval_patch_dir;
    fs::path eval_truth_dir;

    // 检出配方。默认值与各模块原参数一致，实际以 main 中赋值为准。
    HistogramThresholdParams histogram = histogramThresholdDefaultParams();
    EccAlignParams ecc;
    DefectDetectParams defect;
    int overlay_preview_px = kDetectTarget;

    // DieToDie：切 die patch，用邻居 die 做 D2D。
    // DieToDatabase：切 die patch，用一张参考 patch 模板做同样的对比。
    // YoloOpenVino：切 die patch，用 OpenVINO YOLO26 语义分割推理。
    // SingleDie：先按 XML 切并保存全部 die patch，再对圆形 ROI 做本片检出。
    InspectMode mode = InspectMode::DieToDie;

    // YoloOpenVino：best.xml 由 dynamic ONNX 转出，输入 [?,3,64,64]，语义分割。
    fs::path yolo_model;
    std::string yolo_device = "CPU";
    int yolo_imgsz = 64;
    int yolo_batch = 8;
    int yolo_requests = 0;
    float yolo_confidence = 0.25f;
    bool yolo_fp16 = true;
    bool yolo_throughput = false;
    singledie::IntensityNormalizeParams singledie_normalize;
    singledie::DetectionParams singledie_detect;
};

struct Job {
    fs::path image;
    fs::path xml;
    fs::path out;
    std::string name;
};

}  // namespace cutdie
