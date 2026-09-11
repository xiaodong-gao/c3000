#pragma once

#include "die_types.h"

#include <opencv2/core.hpp>

namespace cutdie {

struct CutResult {
    fs::path patch_dir;
    fs::path overlay_path;
    fs::path csv_path;
    int saved = 0;
    int skipped = 0;
    cv::Mat full;                 // 原始输入晶圆图（未缩放）
    std::vector<DieBox> boxes;    // 原图坐标系下的 die 框
};

// 按 XML 从大图切割全部 patch，保存到 opt.patch_out。
// resolution_um：每个像素对应多少微米 (um/px)。
CutResult cut_patches(const Options& opt, double resolution_um);

}  // namespace cutdie
