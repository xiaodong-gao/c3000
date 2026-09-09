#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct AdaptiveThresholdParams {
    int blockSize = 31;   // odd local window; <=0 auto from image size
    double k = 3.0;       // localMean + k * localStd, and robust global median + k * 1.4826 * MAD
    double minAbs = 6.0;  // absolute L2 floor
};

struct MorphologicalParams {
    int openSize = 3;
    int closeSize = 5;
};

struct BlobFilterParams {
    int minArea = 50;
    int maxArea = 20000;  // 0 = unlimited
    int minWidth = 5;
    int minHeight = 5;
    double maxAspectRatio = 12.0;
};

struct DefectDetectParams {
    AdaptiveThresholdParams adaptive;
    MorphologicalParams morph;
    BlobFilterParams blob;
};

struct DefectCandidate {
    int id = 0;
    cv::Rect bbox;
    cv::Point2d centroid;
    int area = 0;
    double meanL2 = 0.0;
    double maxL2 = 0.0;
};

struct DefectDetectOutput {
    bool ok = false;
    std::string error;
    cv::Mat adaptiveMask;  // CV_8UC1, 0/255
    cv::Mat morphMask;
    cv::Mat blobMask;
    cv::Mat overlay;
    std::vector<DefectCandidate> candidates;
};

cv::Mat adaptiveThresholdDiff(const cv::Mat& l2Difference, const AdaptiveThresholdParams& params = {});

cv::Mat morphologicalClean(const cv::Mat& binary, const MorphologicalParams& params = {});

cv::Mat blobFilter(const cv::Mat& binary, const BlobFilterParams& params = {});

std::vector<DefectCandidate> extractDefectCandidates(const cv::Mat& blobMask,
                                                     const cv::Mat& l2Difference);

cv::Mat drawDefectOverlay(const cv::Mat& base, const std::vector<DefectCandidate>& candidates);

DefectDetectOutput detectDefects(const cv::Mat& l2Difference, const cv::Mat& overlayBase,
                                 const DefectDetectParams& params = {});
