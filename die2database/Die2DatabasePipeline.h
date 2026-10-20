#pragma once

#include "die_types.h"

#include <memory>
#include <string>
#include <vector>

namespace cutdie {

// 一张参考模板和一颗切下来的 patch 的对比结果。
// candidates 的坐标在重叠区内，调用方加上 commonRoi 才是 patch 坐标。
struct DatabaseMatch {
    bool ok = false;
    std::string error;
    cv::Rect commonRoi;
    cv::Mat blobMask;
    std::vector<DefectCandidate> candidates;
};

// 不调用 Die-to-Die 流水线。自己完成对位、归一、差分和缺陷筛选。
class DatabaseComparator {
public:
    explicit DatabaseComparator(const Options& opt);
    ~DatabaseComparator();

    DatabaseComparator(const DatabaseComparator&) = delete;
    DatabaseComparator& operator=(const DatabaseComparator&) = delete;

    std::string backendText() const;
    DatabaseMatch match(const cv::Mat& reference, const cv::Mat& patch);
    std::vector<DefectCandidate> findBrightBlocks(const cv::Mat& patch) const;
    std::vector<DefectCandidate> findDarkBlocks(const cv::Mat& patch) const;
    cv::Mat draw(const cv::Mat& patch, const std::vector<DefectCandidate>& defects) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cutdie
