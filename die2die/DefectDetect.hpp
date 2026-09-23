#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct AdaptiveThresholdParams {
    int blockSize = 31;   // odd local window; <=0 auto from image size
    double k = 3.0;       // localMean + k * localStd, and robust global median + k * 1.4826 * MAD
    double minAbs = 6.0;  // absolute L2 floor
};

struct SignedDiffParams {
    double T = 0.0;      // bright polarity: signed > +T；<=0 auto from MAD
    double darkT = 0.0;  // dark polarity: signed < -darkT；<=0 则用 T
};

struct DarkSpotParams {
    bool enableBlackHat = true;  // 黑帽增强，专门抓亮底上的小黑点
    int kernel = 11;             // 应大于目标黑点
    double minResponse = 6.0;    // 黑帽响应下限
};

struct MorphologicalParams {
    int openSize = 3;
    int closeSize = 5;
    int dilateSize = 0;       // <3 skip
    int erodeSize = 0;        // <3 skip
    int removeNoiseSize = 0;  // extra open to drop speckles; <3 skip
    int darkOpenSize = 0;     // 暗通道单独开运算后并入；<3 则跳过。建议 3，避免 openSize 过大吃掉小黑点
};

struct BlobFilterParams {
    int minArea = 50;
    int maxArea = 20000;  // 0 = unlimited
    int minWidth = 5;
    int minHeight = 5;
    double maxAspectRatio = 12.0;
    int borderMargin = 2;  // 贴边长条伪差用；0 关闭。紧凑贴边黑点会保留
    double borderStripAspect = 2.5;  // 贴边且长宽比大于此值才丢（切割/对位长条）
    int minBrightArea = 0; // >0 时亮缺陷用更大面积门限，用来丢掉小亮伪差
};

struct DefectFilterParams {
    bool keepBright = true;
    bool keepDark = true;
    bool keepMixed = true;
    bool keepUnknown = true;
    double minMeanL2 = 0.0;
    double minAbsMeanSigned = 0.0;
};

struct LargeWhiteParams {
    bool enable = false;          // 在本颗 die 上按亮度抓大块白区（不依赖邻居）
    double threshold = 230.0;     // 灰度下限（块状 / 较亮白点）
    int minArea = 4000;           // 排除总线小白块
    int maxArea = 0;              // >0：绝对亮度通路面积上限；0 不限制
    double minFill = 0.15;        // bbox 填充率，排除散点连成的假大块
    double maxAspectRatio = 3.0;  // 排除细长亮条
    int morphSize = 5;           // 开+闭核宽；<3 跳过。小白点应关掉，否则 5x5 会吃掉
    double maxBackground = 0.0; // >0：bbox 外圈均值须 <= 此值，只保留暗底上的白点
    int backgroundMargin = 12;   // 背景外圈宽度
    int tophatSize = 0;          // 白帽核宽；<3 关闭。用来抓暗底上的浅小白点
    double tophatMin = 0.0;      // 白帽响应下限；>0 时另走一条浅点掩膜
    double tophatFloor = 0.0;    // 浅点还须灰度 >= 此值；<=0 则用 50
    int tophatMinArea = 0;       // >0 时白帽通路用独立面积下限
    int tophatMaxArea = 0;       // >0 时白帽通路面积上限
    double tophatMinFill = 0.0;  // >0 时白帽通路填充率；否则用 minFill
    double tophatMaxAspect = 0.0;// >0 时白帽通路长宽比；否则用 maxAspectRatio
    int tophatStreetBand = 0;    // 白帽掩膜挖掉中心划片槽带；0 关闭
    int tophatBorderMargin = 0;  // 白帽丢掉贴边块；0 关闭
    double tophatMinPeak = 0.0;  // 白帽块内灰度峰值下限；0 关闭
    // 圆内孤立小白点（1x1 等）：暗场会被高斯背景吃掉；圆掩膜还会把亮边截成假点。
    // 在未抠圆的原图上找，再与内缩圆相交。isolatedMaxArea<=0 关闭。
    int isolatedMaxArea = 0;
    double isolatedThreshold = 0.0;
    int isolatedInnerMargin = 16;  // 圆掩膜通路丢掉贴边假小点；孤立通路不内缩
    int isolatedNearKeep = 0;      // >0：面积 1 且距边<=此值保留（先过 inner_m）
    int isolatedFarMargin = 0;     // nearKeep>0：1px 距边>=此值保留；nearKeep=0：1px 距边>=此值丢掉
};

struct DarkRegionParams {
    bool enable = false;           // 在本颗 die 上按黑帽抓紧凑暗粒子（不依赖邻居）
    int blackhatSize = 15;         // 黑帽核宽；<3 关闭
    double blackhatMin = 18.0;     // 黑帽响应下限
    double belowMedian = 16.0;     // 像素还须 <= 有效区中值 - 此值，避开网格/划片槽
    double grayMin = 50.0;         // 丢掉片外纯黑
    double grayMax = 220.0;        // 丢掉切割白边
    int minArea = 16;
    int maxArea = 180;
    double minFill = 0.30;
    double maxAspectRatio = 3.5;
    int closeSize = 0;             // >=3：浅点第二通路先闭运算再提块
    double faintBlackhatMin = 0.0; // >0 启用浅点通路
    double faintBelowMedian = 0.0; // 浅点相对中值；<=0 则用 belowMedian
    int faintMinArea = 48;
    double faintMinFill = 0.50;
    double faintMaxAspect = 1.7;
    int streetBand = 0;            // 浅点丢掉中心划片槽带；0 关闭
    int borderMargin = 0;          // 浅点丢掉贴边块；0 关闭
    int faintSmallMinArea = 0;     // >0：再收面积更小但黑帽更强的浅点
    int faintSmallMaxArea = 47;
    double faintSmallMinFill = 0.36;
    double faintSmallMaxAspect = 2.4;
    double faintSmallHatMin = 0.0; // 小浅点块内黑帽峰值下限；<=0 关闭
    int edgeBrightMin = 0;         // >0：白像素数达到此值且黑像素达到 edgeDarkMin 时视为片边 die
    int edgeDarkMin = 0;
    double edgeGrayMin = 90.0;
    double edgeGrayMax = 220.0;
    double edgeBlackhatMin = 14.0;
    double edgeBelowMedian = 10.0;
    int edgeMinArea = 0;           // >0 启用片边浅小点通路（不挖槽、不丢贴边）
    int edgeMaxArea = 150;
    double edgeMinFill = 0.30;
    double edgeMaxAspect = 3.5;
    int edgeNearCut = 0;           // >0：只保留距切割白边这么多像素内的浅点
    double edgeBlackFrac = 0.65;   // 黑像素须集中在图像上/下 45%，排除左右片边
};

struct DefectDetectParams {
    AdaptiveThresholdParams adaptive;
    SignedDiffParams signedDiff;
    DarkSpotParams darkSpot;
    MorphologicalParams morph;
    BlobFilterParams blob;
    DefectFilterParams filter;
    LargeWhiteParams largeWhite;
    DarkRegionParams darkRegion;
};

enum class DefectPolarity { Unknown, Bright, Dark, Mixed };

struct DefectCandidate {
    int id = 0;
    int labelId = 0;
    cv::Rect bbox;
    cv::Point2d centroid;
    int area = 0;
    double meanL2 = 0.0;
    double maxL2 = 0.0;
    double meanSigned = 0.0;
    double maxSigned = 0.0;
    double minSigned = 0.0;
    int brightArea = 0;
    int darkArea = 0;
    DefectPolarity polarity = DefectPolarity::Unknown;
};

struct DefectDetectOutput {
    bool ok = false;
    std::string error;
    cv::Mat signedDifference;  // CV_32FC1, dataNormalized - templateNormalized
    cv::Mat fusedDifference;   // L2 gated by bright/dark polarity
    cv::Mat brightMask;        // signed > +T
    cv::Mat darkMask;          // signed < -T
    cv::Mat polarityMask;      // bright | dark
    cv::Mat adaptiveMask;      // CV_8UC1, 0/255
    cv::Mat morphMask;
    cv::Mat blobMask;
    cv::Mat overlay;
    double polarityT = 0.0;
    double polarityDarkT = 0.0;
    std::vector<DefectCandidate> candidates;
};

const char* defectPolarityName(DefectPolarity polarity);

cv::Mat signedDifferenceMap(const cv::Mat& templateNormalized, const cv::Mat& dataNormalized);

cv::Mat polarityMasks(const cv::Mat& signedDifference, double brightT, double darkT,
                      cv::Mat& brightMask, cv::Mat& darkMask);

cv::Mat polarityMasks(const cv::Mat& signedDifference, double T, cv::Mat& brightMask,
                      cv::Mat& darkMask);

double signedPolarityThreshold(const cv::Mat& signedDifference, const DefectDetectParams& params);

double signedDarkPolarityThreshold(const DefectDetectParams& params, double brightT);

cv::Mat darkSpotAssistMask(const cv::Mat& dataNormalized, const cv::Mat& signedDifference,
                           const DarkSpotParams& params, double signedGate);

void clearMaskBorder(cv::Mat& mask, int margin);

cv::Mat fuseDifference(const cv::Mat& l2Difference, const cv::Mat& polarityMask);

void applyDefectPolarity(const cv::Mat& l2Difference, const cv::Mat& templateNormalized,
                         const cv::Mat& dataNormalized, const DefectDetectParams& params,
                         cv::Mat& signedDifference, cv::Mat& brightMask, cv::Mat& darkMask,
                         cv::Mat& polarityMask, cv::Mat& fusedDifference, double& polarityT,
                         double& polarityDarkT);

cv::Mat mergeMorphWithDarkChannel(const cv::Mat& adaptiveMask, const cv::Mat& darkMask,
                                  const MorphologicalParams& morph);

cv::Mat adaptiveThresholdDiff(const cv::Mat& l2Difference, const AdaptiveThresholdParams& params = {});

cv::Mat morphologicalClean(const cv::Mat& binary, const MorphologicalParams& params = {});

cv::Mat blobFilter(const cv::Mat& binary, const BlobFilterParams& params = {});

std::vector<DefectCandidate> extractDefectCandidates(const cv::Mat& blobMask,
                                                     const cv::Mat& l2Difference,
                                                     const cv::Mat& signedDifference = {},
                                                     const cv::Mat& brightMask = {},
                                                     const cv::Mat& darkMask = {});

std::vector<DefectCandidate> extractLargeWhiteRegions(const cv::Mat& image,
                                                      const LargeWhiteParams& params);

std::vector<DefectCandidate> extractDarkRegions(const cv::Mat& image,
                                                const DarkRegionParams& params);

std::vector<DefectCandidate> filterDefects(const std::vector<DefectCandidate>& candidates,
                                           const BlobFilterParams& blob,
                                           const DefectFilterParams& filter = {},
                                           cv::Size imageSize = {});

cv::Mat maskFromCandidates(const cv::Size& size, const std::vector<DefectCandidate>& candidates,
                           const cv::Mat& labels);

cv::Mat drawDefectOverlay(const cv::Mat& base, const std::vector<DefectCandidate>& candidates);

DefectDetectOutput detectDefects(const cv::Mat& l2Difference, const cv::Mat& overlayBase,
                                 const DefectDetectParams& params = {},
                                 const cv::Mat& templateNormalized = {},
                                 const cv::Mat& dataNormalized = {});
