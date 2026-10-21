#include "recipe_json.h"

#include "json.hpp"

#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::json;

namespace {

template <typename T>
void read_field(const json& j, const char* key, T& dst) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return;
    }
    dst = it->get<T>();
}

template <typename Enum, size_t N>
void read_enum(const json& j, const char* key, Enum& dst,
               const std::pair<const char*, Enum> (&table)[N]) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return;
    }
    if (!it->is_string()) {
        throw std::runtime_error(std::string(key) + " 必须是字符串");
    }
    const std::string name = it->get<std::string>();
    for (const auto& item : table) {
        if (name == item.first) {
            dst = item.second;
            return;
        }
    }
    throw std::runtime_error(std::string(key) + " 无法识别: " + name);
}

template <typename Enum, size_t N>
const char* enum_name(Enum value, const std::pair<const char*, Enum> (&table)[N]) {
    for (const auto& item : table) {
        if (item.second == value) {
            return item.first;
        }
    }
    throw std::runtime_error("无法写出未知枚举");
}

void reject_unknown(const json& j, std::initializer_list<const char*> keys, const char* owner) {
    if (!j.is_object()) {
        throw std::runtime_error(std::string(owner) + " 必须是 JSON 对象");
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        bool known = false;
        for (const char* name : keys) {
            if (it.key() == name) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw std::runtime_error(std::string(owner) + " 含有未知字段: " + it.key());
        }
    }
}

const std::pair<const char*, cutdie::InspectMode> kInspectModes[] = {
    {"SingleDie", cutdie::InspectMode::SingleDie},
    {"DieToDie", cutdie::InspectMode::DieToDie},
    {"DieToDatabase", cutdie::InspectMode::DieToDatabase},
    {"YoloOpenVino", cutdie::InspectMode::YoloOpenVino},
    {"AnomalibOpenVino", cutdie::InspectMode::AnomalibOpenVino},
};

const std::pair<const char*, cutdie::AlignMethod> kAlignMethods[] = {
    {"PhaseCuda", cutdie::AlignMethod::PhaseCuda},
    {"PhaseCpu", cutdie::AlignMethod::PhaseCpu},
    {"EccCuda", cutdie::AlignMethod::EccCuda},
    {"EccCpu", cutdie::AlignMethod::EccCpu},
};

const std::pair<const char*, cutdie::HistogramMethod> kHistogramMethods[] = {
    {"Auto", cutdie::HistogramMethod::Auto},
    {"Cpu", cutdie::HistogramMethod::Cpu},
    {"Cuda", cutdie::HistogramMethod::Cuda},
};

const std::pair<const char*, cutdie::NormalizeMethod> kNormalizeMethods[] = {
    {"Auto", cutdie::NormalizeMethod::Auto},
    {"Cpu", cutdie::NormalizeMethod::Cpu},
    {"Cuda", cutdie::NormalizeMethod::Cuda},
};

const std::pair<const char*, singledie::BackgroundMethod> kBackgroundMethods[] = {
    {"Gaussian", singledie::BackgroundMethod::Gaussian},
    {"Median", singledie::BackgroundMethod::Median},
    {"MorphOpen", singledie::BackgroundMethod::MorphOpen},
};

const std::pair<const char*, singledie::IntensityMethod> kIntensityMethods[] = {
    {"Divide", singledie::IntensityMethod::Divide},
    {"Subtract", singledie::IntensityMethod::Subtract},
    {"SubtractDivide", singledie::IntensityMethod::SubtractDivide},
};

json motion_to_json(int motion) {
    if (motion == cv::MOTION_TRANSLATION) {
        return "TRANSLATION";
    }
    if (motion == cv::MOTION_EUCLIDEAN) {
        return "EUCLIDEAN";
    }
    if (motion == cv::MOTION_AFFINE) {
        return "AFFINE";
    }
    if (motion == cv::MOTION_HOMOGRAPHY) {
        return "HOMOGRAPHY";
    }
    return motion;
}

void read_motion(const json& j, int& motion) {
    const auto it = j.find("motionType");
    if (it == j.end() || it->is_null()) {
        return;
    }
    if (it->is_number_integer()) {
        motion = it->get<int>();
        return;
    }
    if (!it->is_string()) {
        throw std::runtime_error("motionType 必须是字符串或整数");
    }
    const std::string name = it->get<std::string>();
    if (name == "TRANSLATION" || name == "MOTION_TRANSLATION") {
        motion = cv::MOTION_TRANSLATION;
    } else if (name == "EUCLIDEAN" || name == "MOTION_EUCLIDEAN") {
        motion = cv::MOTION_EUCLIDEAN;
    } else if (name == "AFFINE" || name == "MOTION_AFFINE") {
        motion = cv::MOTION_AFFINE;
    } else if (name == "HOMOGRAPHY" || name == "MOTION_HOMOGRAPHY") {
        motion = cv::MOTION_HOMOGRAPHY;
    } else {
        throw std::runtime_error("motionType 无法识别: " + name);
    }
}

json dies_to_json(const std::set<std::pair<int, int>>& dies) {
    json out = json::array();
    for (const auto& xy : dies) {
        out.push_back(json::array({xy.first, xy.second}));
    }
    return out;
}

void read_dies(const json& j, const char* key, std::set<std::pair<int, int>>& dst) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return;
    }
    if (!it->is_array()) {
        throw std::runtime_error(std::string(key) + " 必须是 [[x, y], ...]");
    }
    std::set<std::pair<int, int>> parsed;
    for (const auto& item : *it) {
        if (!item.is_array() || item.size() != 2 || !item.at(0).is_number_integer() ||
            !item.at(1).is_number_integer()) {
            throw std::runtime_error(std::string(key) + " 的每一项必须是 [x, y]");
        }
        parsed.emplace(item.at(0).get<int>(), item.at(1).get<int>());
    }
    dst = std::move(parsed);
}

json path_to_json(const std::filesystem::path& path) {
    return path.generic_string();
}

void read_path(const json& j, const char* key, std::filesystem::path& dst) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return;
    }
    if (!it->is_string()) {
        throw std::runtime_error(std::string(key) + " 必须是字符串");
    }
    dst = std::filesystem::path(it->get<std::string>());
}

#define READ(name) read_field(j, #name, p.name)

}  // namespace

void to_json(json& j, const AdaptiveThresholdParams& p) {
    j = json{{"blockSize", p.blockSize}, {"k", p.k}, {"minAbs", p.minAbs}};
}

void from_json(const json& j, AdaptiveThresholdParams& p) {
    reject_unknown(j, {"blockSize", "k", "minAbs"}, "adaptive");
    READ(blockSize);
    READ(k);
    READ(minAbs);
}

void to_json(json& j, const SignedDiffParams& p) {
    j = json{{"T", p.T}, {"darkT", p.darkT}};
}

void from_json(const json& j, SignedDiffParams& p) {
    reject_unknown(j, {"T", "darkT"}, "signedDiff");
    READ(T);
    READ(darkT);
}

void to_json(json& j, const DarkSpotParams& p) {
    j = json{{"enableBlackHat", p.enableBlackHat}, {"kernel", p.kernel}, {"minResponse", p.minResponse}};
}

void from_json(const json& j, DarkSpotParams& p) {
    reject_unknown(j, {"enableBlackHat", "kernel", "minResponse"}, "darkSpot");
    READ(enableBlackHat);
    READ(kernel);
    READ(minResponse);
}

void to_json(json& j, const MorphologicalParams& p) {
    j = json{{"openSize", p.openSize},         {"closeSize", p.closeSize},
             {"dilateSize", p.dilateSize},     {"erodeSize", p.erodeSize},
             {"removeNoiseSize", p.removeNoiseSize}, {"darkOpenSize", p.darkOpenSize}};
}

void from_json(const json& j, MorphologicalParams& p) {
    reject_unknown(j, {"openSize", "closeSize", "dilateSize", "erodeSize", "removeNoiseSize", "darkOpenSize"},
                   "morph");
    READ(openSize);
    READ(closeSize);
    READ(dilateSize);
    READ(erodeSize);
    READ(removeNoiseSize);
    READ(darkOpenSize);
}

void to_json(json& j, const BlobFilterParams& p) {
    j = json{{"minArea", p.minArea},
             {"maxArea", p.maxArea},
             {"minWidth", p.minWidth},
             {"minHeight", p.minHeight},
             {"maxAspectRatio", p.maxAspectRatio},
             {"borderMargin", p.borderMargin},
             {"borderStripAspect", p.borderStripAspect},
             {"minBrightArea", p.minBrightArea}};
}

void from_json(const json& j, BlobFilterParams& p) {
    reject_unknown(j,
                   {"minArea", "maxArea", "minWidth", "minHeight", "maxAspectRatio", "borderMargin",
                    "borderStripAspect", "minBrightArea"},
                   "blob");
    READ(minArea);
    READ(maxArea);
    READ(minWidth);
    READ(minHeight);
    READ(maxAspectRatio);
    READ(borderMargin);
    READ(borderStripAspect);
    READ(minBrightArea);
}

void to_json(json& j, const DefectFilterParams& p) {
    j = json{{"keepBright", p.keepBright},
             {"keepDark", p.keepDark},
             {"keepMixed", p.keepMixed},
             {"keepUnknown", p.keepUnknown},
             {"minMeanL2", p.minMeanL2},
             {"minAbsMeanSigned", p.minAbsMeanSigned}};
}

void from_json(const json& j, DefectFilterParams& p) {
    reject_unknown(j,
                   {"keepBright", "keepDark", "keepMixed", "keepUnknown", "minMeanL2", "minAbsMeanSigned"},
                   "filter");
    READ(keepBright);
    READ(keepDark);
    READ(keepMixed);
    READ(keepUnknown);
    READ(minMeanL2);
    READ(minAbsMeanSigned);
}

void to_json(json& j, const LargeWhiteParams& p) {
    j = json{{"enable", p.enable},
             {"threshold", p.threshold},
             {"minArea", p.minArea},
             {"maxArea", p.maxArea},
             {"minFill", p.minFill},
             {"maxAspectRatio", p.maxAspectRatio},
             {"morphSize", p.morphSize},
             {"maxBackground", p.maxBackground},
             {"backgroundMargin", p.backgroundMargin},
             {"tophatSize", p.tophatSize},
             {"tophatMin", p.tophatMin},
             {"tophatFloor", p.tophatFloor},
             {"tophatMinArea", p.tophatMinArea},
             {"tophatMaxArea", p.tophatMaxArea},
             {"tophatMinFill", p.tophatMinFill},
             {"tophatMaxAspect", p.tophatMaxAspect},
             {"tophatStreetBand", p.tophatStreetBand},
             {"tophatBorderMargin", p.tophatBorderMargin},
             {"tophatMinPeak", p.tophatMinPeak},
             {"isolatedMaxArea", p.isolatedMaxArea},
             {"isolatedThreshold", p.isolatedThreshold},
             {"isolatedInnerMargin", p.isolatedInnerMargin},
             {"isolatedNearKeep", p.isolatedNearKeep},
             {"isolatedFarMargin", p.isolatedFarMargin}};
}

void from_json(const json& j, LargeWhiteParams& p) {
    reject_unknown(j,
                   {"enable", "threshold", "minArea", "maxArea", "minFill", "maxAspectRatio", "morphSize",
                    "maxBackground", "backgroundMargin", "tophatSize", "tophatMin", "tophatFloor",
                    "tophatMinArea", "tophatMaxArea", "tophatMinFill", "tophatMaxAspect", "tophatStreetBand",
                    "tophatBorderMargin", "tophatMinPeak", "isolatedMaxArea", "isolatedThreshold",
                    "isolatedInnerMargin", "isolatedNearKeep", "isolatedFarMargin"},
                   "largeWhite");
    READ(enable);
    READ(threshold);
    READ(minArea);
    READ(maxArea);
    READ(minFill);
    READ(maxAspectRatio);
    READ(morphSize);
    READ(maxBackground);
    READ(backgroundMargin);
    READ(tophatSize);
    READ(tophatMin);
    READ(tophatFloor);
    READ(tophatMinArea);
    READ(tophatMaxArea);
    READ(tophatMinFill);
    READ(tophatMaxAspect);
    READ(tophatStreetBand);
    READ(tophatBorderMargin);
    READ(tophatMinPeak);
    READ(isolatedMaxArea);
    READ(isolatedThreshold);
    READ(isolatedInnerMargin);
    READ(isolatedNearKeep);
    READ(isolatedFarMargin);
}

void to_json(json& j, const DarkRegionParams& p) {
    j = json{{"enable", p.enable},
             {"blackhatSize", p.blackhatSize},
             {"blackhatMin", p.blackhatMin},
             {"belowMedian", p.belowMedian},
             {"grayMin", p.grayMin},
             {"grayMax", p.grayMax},
             {"minArea", p.minArea},
             {"maxArea", p.maxArea},
             {"minFill", p.minFill},
             {"maxAspectRatio", p.maxAspectRatio},
             {"closeSize", p.closeSize},
             {"faintBlackhatMin", p.faintBlackhatMin},
             {"faintBelowMedian", p.faintBelowMedian},
             {"faintMinArea", p.faintMinArea},
             {"faintMinFill", p.faintMinFill},
             {"faintMaxAspect", p.faintMaxAspect},
             {"streetBand", p.streetBand},
             {"borderMargin", p.borderMargin},
             {"faintSmallMinArea", p.faintSmallMinArea},
             {"faintSmallMaxArea", p.faintSmallMaxArea},
             {"faintSmallMinFill", p.faintSmallMinFill},
             {"faintSmallMaxAspect", p.faintSmallMaxAspect},
             {"faintSmallHatMin", p.faintSmallHatMin},
             {"edgeBrightMin", p.edgeBrightMin},
             {"edgeDarkMin", p.edgeDarkMin},
             {"edgeGrayMin", p.edgeGrayMin},
             {"edgeGrayMax", p.edgeGrayMax},
             {"edgeBlackhatMin", p.edgeBlackhatMin},
             {"edgeBelowMedian", p.edgeBelowMedian},
             {"edgeMinArea", p.edgeMinArea},
             {"edgeMaxArea", p.edgeMaxArea},
             {"edgeMinFill", p.edgeMinFill},
             {"edgeMaxAspect", p.edgeMaxAspect},
             {"edgeNearCut", p.edgeNearCut},
             {"edgeBlackFrac", p.edgeBlackFrac}};
}

void from_json(const json& j, DarkRegionParams& p) {
    reject_unknown(j,
                   {"enable", "blackhatSize", "blackhatMin", "belowMedian", "grayMin", "grayMax", "minArea",
                    "maxArea", "minFill", "maxAspectRatio", "closeSize", "faintBlackhatMin", "faintBelowMedian",
                    "faintMinArea", "faintMinFill", "faintMaxAspect", "streetBand", "borderMargin",
                    "faintSmallMinArea", "faintSmallMaxArea", "faintSmallMinFill", "faintSmallMaxAspect",
                    "faintSmallHatMin", "edgeBrightMin", "edgeDarkMin", "edgeGrayMin", "edgeGrayMax",
                    "edgeBlackhatMin", "edgeBelowMedian", "edgeMinArea", "edgeMaxArea", "edgeMinFill",
                    "edgeMaxAspect", "edgeNearCut", "edgeBlackFrac"},
                   "darkRegion");
    READ(enable);
    READ(blackhatSize);
    READ(blackhatMin);
    READ(belowMedian);
    READ(grayMin);
    READ(grayMax);
    READ(minArea);
    READ(maxArea);
    READ(minFill);
    READ(maxAspectRatio);
    READ(closeSize);
    READ(faintBlackhatMin);
    READ(faintBelowMedian);
    READ(faintMinArea);
    READ(faintMinFill);
    READ(faintMaxAspect);
    READ(streetBand);
    READ(borderMargin);
    READ(faintSmallMinArea);
    READ(faintSmallMaxArea);
    READ(faintSmallMinFill);
    READ(faintSmallMaxAspect);
    READ(faintSmallHatMin);
    READ(edgeBrightMin);
    READ(edgeDarkMin);
    READ(edgeGrayMin);
    READ(edgeGrayMax);
    READ(edgeBlackhatMin);
    READ(edgeBelowMedian);
    READ(edgeMinArea);
    READ(edgeMaxArea);
    READ(edgeMinFill);
    READ(edgeMaxAspect);
    READ(edgeNearCut);
    READ(edgeBlackFrac);
}

void to_json(json& j, const DefectDetectParams& p) {
    j = json{{"adaptive", p.adaptive},   {"signedDiff", p.signedDiff}, {"darkSpot", p.darkSpot},
             {"morph", p.morph},         {"blob", p.blob},             {"filter", p.filter},
             {"largeWhite", p.largeWhite}, {"darkRegion", p.darkRegion}};
}

void from_json(const json& j, DefectDetectParams& p) {
    reject_unknown(j,
                   {"adaptive", "signedDiff", "darkSpot", "morph", "blob", "filter", "largeWhite", "darkRegion"},
                   "defect");
    READ(adaptive);
    READ(signedDiff);
    READ(darkSpot);
    READ(morph);
    READ(blob);
    READ(filter);
    READ(largeWhite);
    READ(darkRegion);
}

void to_json(json& j, const EccAlignParams& p) {
    j = json{{"maxIters", p.maxIters},
             {"eps", p.eps},
             {"gaussFiltSize", p.gaussFiltSize},
             {"motionType", motion_to_json(p.motionType)}};
}

void from_json(const json& j, EccAlignParams& p) {
    reject_unknown(j, {"maxIters", "eps", "gaussFiltSize", "motionType"}, "ecc");
    READ(maxIters);
    READ(eps);
    READ(gaussFiltSize);
    read_motion(j, p.motionType);
}

void to_json(json& j, const HistogramThresholdParams& p) {
    j = json{{"gaussianSigma", p.gaussianSigma},
             {"peakHeightRatio", p.peakHeightRatio},
             {"minPeakDistance", p.minPeakDistance},
             {"minPeakCountRatio", p.minPeakCountRatio},
             {"defaultThreshold", p.defaultThreshold}};
}

void from_json(const json& j, HistogramThresholdParams& p) {
    p = histogramThresholdDefaultParams();
    reject_unknown(j,
                   {"gaussianSigma", "peakHeightRatio", "minPeakDistance", "minPeakCountRatio", "defaultThreshold"},
                   "histogram");
    READ(gaussianSigma);
    READ(peakHeightRatio);
    READ(minPeakDistance);
    READ(minPeakCountRatio);
    READ(defaultThreshold);
}

namespace singledie {

void to_json(json& j, const BackgroundParams& p) {
    j = json{{"method", enum_name(p.method, kBackgroundMethods)},
             {"kernelSize", p.kernelSize},
             {"sigma", p.sigma}};
}

void from_json(const json& j, BackgroundParams& p) {
    reject_unknown(j, {"method", "kernelSize", "sigma"}, "background");
    read_enum(j, "method", p.method, kBackgroundMethods);
    READ(kernelSize);
    READ(sigma);
}

void to_json(json& j, const IntensityNormalizeParams& p) {
    j = json{{"background", p.background},
             {"method", enum_name(p.method, kIntensityMethods)},
             {"minBackground", p.minBackground},
             {"robustPercentile", p.robustPercentile},
             {"lowPercentile", p.lowPercentile},
             {"highPercentile", p.highPercentile},
             {"subtractCenter", p.subtractCenter}};
}

void from_json(const json& j, IntensityNormalizeParams& p) {
    reject_unknown(j,
                   {"background", "method", "minBackground", "robustPercentile", "lowPercentile", "highPercentile",
                    "subtractCenter"},
                   "singledie_normalize");
    READ(background);
    read_enum(j, "method", p.method, kIntensityMethods);
    READ(minBackground);
    READ(robustPercentile);
    READ(lowPercentile);
    READ(highPercentile);
    READ(subtractCenter);
}

void to_json(json& j, const DetectionParams& p) {
    j = json{{"brightThreshold", p.brightThreshold},
             {"darkThreshold", p.darkThreshold},
             {"morphologyKernel", p.morphologyKernel},
             {"openIterations", p.openIterations},
             {"closeIterations", p.closeIterations},
             {"minArea", p.minArea},
             {"maxArea", p.maxArea},
             {"borderMargin", p.borderMargin},
             {"enableBright", p.enableBright},
             {"enableDark", p.enableDark},
             {"keepResponseMaps", p.keepResponseMaps},
             {"enableBlob", p.enableBlob},
             {"useAbsoluteBright", p.useAbsoluteBright},
             {"maxAspectRatio", p.maxAspectRatio},
             {"minFillRatio", p.minFillRatio},
             {"normalReference", p.normalReference}};
}

void from_json(const json& j, DetectionParams& p) {
    reject_unknown(j,
                   {"brightThreshold", "darkThreshold", "morphologyKernel", "openIterations", "closeIterations",
                    "minArea", "maxArea", "borderMargin", "enableBright", "enableDark", "keepResponseMaps",
                    "enableBlob", "useAbsoluteBright", "maxAspectRatio", "minFillRatio", "normalReference"},
                   "singledie_detect");
    READ(brightThreshold);
    READ(darkThreshold);
    READ(morphologyKernel);
    READ(openIterations);
    READ(closeIterations);
    READ(minArea);
    READ(maxArea);
    READ(borderMargin);
    READ(enableBright);
    READ(enableDark);
    READ(keepResponseMaps);
    READ(enableBlob);
    READ(useAbsoluteBright);
    READ(maxAspectRatio);
    READ(minFillRatio);
    READ(normalReference);
}

}  // namespace singledie

namespace cutdie {

void to_json(json& j, const Options& opt) {
    j = json{{"mode", enum_name(opt.mode, kInspectModes)},
             {"dir", path_to_json(opt.dir)},
             {"image", path_to_json(opt.image)},
             {"xml", path_to_json(opt.xml)},
             {"out", path_to_json(opt.out)},
             {"patch_out", path_to_json(opt.patch_out)},
             {"resolution_um", opt.resolution_um},
             {"keep_skipped", opt.keep_skipped},
             {"skip_inspect", dies_to_json(opt.skip_inspect)},
             {"inspect_only", dies_to_json(opt.inspect_only)},
             {"recursive", opt.recursive},
             {"overlay_preview_px", opt.overlay_preview_px},
             {"template_path", path_to_json(opt.template_path)},
             {"template_neighbor", opt.template_neighbor},
             {"template_neighbor_min", opt.template_neighbor_min},
             {"inspect", opt.inspect},
             {"eval_patch_dir", path_to_json(opt.eval_patch_dir)},
             {"eval_truth_dir", path_to_json(opt.eval_truth_dir)},
             {"align_method", enum_name(opt.align_method, kAlignMethods)},
             {"histogram_method", enum_name(opt.histogram_method, kHistogramMethods)},
             {"normalize_method", enum_name(opt.normalize_method, kNormalizeMethods)},
             {"histogram", opt.histogram},
             {"ecc", opt.ecc},
             {"defect", opt.defect},
             {"singledie_normalize", opt.singledie_normalize},
             {"singledie_detect", opt.singledie_detect},
             {"yolo_model", path_to_json(opt.yolo_model)},
             {"yolo_device", opt.yolo_device},
             {"yolo_imgsz", opt.yolo_imgsz},
             {"yolo_batch", opt.yolo_batch},
             {"yolo_requests", opt.yolo_requests},
             {"yolo_confidence", opt.yolo_confidence},
             {"yolo_fp16", opt.yolo_fp16},
             {"yolo_throughput", opt.yolo_throughput}};
}

void from_json(const json& j, Options& opt) {
    reject_unknown(j,
                   {"mode", "dir", "image", "xml", "out", "patch_out", "resolution_um", "keep_skipped",
                    "skip_inspect", "inspect_only", "recursive", "overlay_preview_px", "template_path",
                    "template_neighbor", "template_neighbor_min", "inspect", "eval_patch_dir", "eval_truth_dir",
                    "align_method", "histogram_method", "normalize_method", "histogram", "ecc", "defect",
                    "singledie_normalize", "singledie_detect", "yolo_model", "yolo_device", "yolo_imgsz",
                    "yolo_batch", "yolo_requests", "yolo_confidence", "yolo_fp16", "yolo_throughput"},
                   "配方");
    if (!j.contains("mode")) {
        throw std::runtime_error("配方缺少 mode");
    }
    read_enum(j, "mode", opt.mode, kInspectModes);
    read_path(j, "dir", opt.dir);
    read_path(j, "image", opt.image);
    read_path(j, "xml", opt.xml);
    read_path(j, "out", opt.out);
    read_path(j, "patch_out", opt.patch_out);
    read_field(j, "resolution_um", opt.resolution_um);
    read_field(j, "keep_skipped", opt.keep_skipped);
    read_dies(j, "skip_inspect", opt.skip_inspect);
    read_dies(j, "inspect_only", opt.inspect_only);
    read_field(j, "recursive", opt.recursive);
    read_field(j, "overlay_preview_px", opt.overlay_preview_px);
    read_path(j, "template_path", opt.template_path);
    read_field(j, "template_neighbor", opt.template_neighbor);
    read_field(j, "template_neighbor_min", opt.template_neighbor_min);
    read_field(j, "inspect", opt.inspect);
    read_path(j, "eval_patch_dir", opt.eval_patch_dir);
    read_path(j, "eval_truth_dir", opt.eval_truth_dir);
    read_enum(j, "align_method", opt.align_method, kAlignMethods);
    read_enum(j, "histogram_method", opt.histogram_method, kHistogramMethods);
    read_enum(j, "normalize_method", opt.normalize_method, kNormalizeMethods);
    read_field(j, "histogram", opt.histogram);
    read_field(j, "ecc", opt.ecc);
    read_field(j, "defect", opt.defect);
    read_field(j, "singledie_normalize", opt.singledie_normalize);
    read_field(j, "singledie_detect", opt.singledie_detect);
    read_path(j, "yolo_model", opt.yolo_model);
    read_field(j, "yolo_device", opt.yolo_device);
    read_field(j, "yolo_imgsz", opt.yolo_imgsz);
    read_field(j, "yolo_batch", opt.yolo_batch);
    read_field(j, "yolo_requests", opt.yolo_requests);
    read_field(j, "yolo_confidence", opt.yolo_confidence);
    read_field(j, "yolo_fp16", opt.yolo_fp16);
    read_field(j, "yolo_throughput", opt.yolo_throughput);
}

namespace {

std::filesystem::path executable_dir() {
#ifdef _WIN32
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0 || n >= buf.size()) {
        return {};
    }
    buf.resize(n);
    return std::filesystem::path(buf).parent_path();
#else
    return {};
#endif
}

const char* recipe_filename(InspectMode mode) {
    switch (mode) {
    case InspectMode::SingleDie:
        return "SingleDie.json";
    case InspectMode::DieToDie:
        return "DieToDie.json";
    case InspectMode::DieToDatabase:
        return "DieToDatabase.json";
    case InspectMode::YoloOpenVino:
    case InspectMode::AnomalibOpenVino:
        return nullptr;
    }
    return nullptr;
}

bool find_up(std::filesystem::path dir, const char* file, std::filesystem::path& found) {
    for (int i = 0; i < 6 && !dir.empty(); ++i) {
        const std::filesystem::path candidate = dir / "recipes" / file;
        if (std::filesystem::is_regular_file(candidate)) {
            found = candidate;
            return true;
        }
        const std::filesystem::path parent = dir.parent_path();
        if (parent == dir) {
            break;
        }
        dir = parent;
    }
    return false;
}

}  // namespace

fs::path find_recipe(InspectMode mode) {
    const char* file = recipe_filename(mode);
    if (!file) {
        throw std::runtime_error("该检出模式没有 JSON 配方");
    }
    fs::path found;
    if (find_up(fs::current_path(), file, found)) {
        return found;
    }
    const fs::path exe = executable_dir();
    if (!exe.empty() && find_up(exe, file, found)) {
        return found;
    }
    throw std::runtime_error(std::string("找不到配方文件 recipes/") + file);
}

Options load_recipe(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("无法打开配方: " + path.string());
    }
    try {
        const json j = json::parse(in, nullptr, true, true);
        if (!j.is_object()) {
            throw std::runtime_error("配方必须是 JSON 对象");
        }
        Options opt;
        from_json(j, opt);
        return opt;
    } catch (const std::exception& ex) {
        throw std::runtime_error("配方 " + path.string() + ": " + ex.what());
    }
}

void save_recipe(const fs::path& path, const Options& opt) {
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("无法写配方: " + path.string());
    }
    const json j = opt;
    out << j.dump(2) << '\n';
    if (!out) {
        throw std::runtime_error("写入配方失败: " + path.string());
    }
}

}  // namespace cutdie

#undef READ
