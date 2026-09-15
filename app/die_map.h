#pragma once

#include "die_types.h"

#include <opencv2/core.hpp>

namespace cutdie {

DieMap parse_die_map(const fs::path& xml_path);

// 按对齐配置得到原图像素坐标系下的物理晶圆圆（半径 = WaferDiameter/2）。
// EdgeOffset / TrimSize 不缩小该圆；Trim CenterX/Y 只平移圆心。
WaferCircle make_wafer_circle(const DieMap& xml, int img_w, int img_h, double resolution_um, int lut);

double infer_resolution_um(const fs::path& path, double fallback = 10.0);

int detect_lut_width(const cv::Mat& gray);

// 按 C3000 DieMapLayout.CreateLayoutPoints / cut_die.py 生成 die 框。
// img_w / img_h 为去掉左侧 LUT 后的晶圆区域尺寸，x_offset 为 LUT 宽度。
std::vector<DieBox> generate_dies(
    const DieMap& xml,
    int img_w,
    int img_h,
    double resolution_um,
    int x_offset);

std::vector<DieBox> scale_boxes(const std::vector<DieBox>& boxes, double scale);

}  // namespace cutdie
