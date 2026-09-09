#pragma once

#include "die_cut.h"

#include <map>
#include <utility>

#include <opencv2/core.hpp>

namespace cutdie {

struct WaferDefect {
    int die_x = 0;
    int die_y = 0;
    int id = 0;
    cv::Rect bbox;  // 原始晶圆图坐标
    cv::Point2d centroid;
    int area = 0;
    double meanL2 = 0.0;
    double maxL2 = 0.0;
};

struct InspectResult {
    fs::path overlay_path;
    fs::path csv_path;
    int inspected = 0;
    int failed = 0;
    int skipped_no_template = 0;
    int skipped_die = 0;
    int defect_count = 0;
    std::vector<WaferDefect> defects;
};

struct TemplateCache {
    bool single = false;
    fs::path source;
    cv::Mat single_img;
    std::map<std::pair<int, int>, cv::Mat> by_xy;
};

// 切割前先读入外部模板，避免 patch 目录被清空后找不到模板。
TemplateCache load_templates(const fs::path& template_path);

// 用外部模板对每个 die 做完整 D2D 检出，把缺陷画回原始晶圆大图并保存。
InspectResult inspect_wafer(const Options& opt, const CutResult& cut, const TemplateCache& templates);

}  // namespace cutdie
