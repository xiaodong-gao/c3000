#pragma once

#include <filesystem>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cutdie {

namespace fs = std::filesystem;

inline constexpr int kDetectTarget = 8000;
inline constexpr const char* kDefaultDir = "C:/Users/aspiring135/Desktop/datasets/10um-gray";

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
    std::set<std::pair<int, int>> skipped;
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
    bool recursive = false;
    double resolution_um = 10.0;  // 每个像素对应多少微米 (um/px)

    // 切割并保存全部 patch。空则写到 <out>/patches
    fs::path patch_out;
};

struct Job {
    fs::path image;
    fs::path xml;
    fs::path out;
    std::string name;
};

}  // namespace cutdie
