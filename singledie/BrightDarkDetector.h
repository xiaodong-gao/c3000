#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace singledie {

struct Defect {
    int id = -1;
    bool bright = true;
    cv::Rect bbox;
    int area = 0;
    cv::Point2d center;
    double meanResponse = 0.0;
    double maxResponse = 0.0;
};

struct DetectionParams {
    float brightThreshold = 0.08f;
    float darkThreshold = 0.08f;

    int morphologyKernel = 3;
    int openIterations = 1;
    int closeIterations = 1;

    int minArea = 5;
    int maxArea = 1000000;

    // Ignore components that touch this many pixels of the image border.
    int borderMargin = 0;

    bool enableBright = true;
    bool enableDark = true;

    // Disable when only defect objects/masks are needed (for example, in a
    // production or benchmark path) to avoid retaining two float images.
    bool keepResponseMaps = true;

    // Compact circular dark-particle detector. When enabled, dark defects are
    // extracted with SimpleBlobDetector instead of thresholded connected
    // components. Bright detection is unchanged.
    bool enableBlob = false;

    // Threshold the source gray image directly instead of the normalized
    // bright response. Used for white bubbles on a dark field.
    bool useAbsoluteBright = false;

    // 0 disables the filter. Otherwise reject components whose bounding-box
    // aspect ratio max(w,h)/min(w,h) exceeds this value.
    float maxAspectRatio = 0.0f;

    // 0 disables the filter. Otherwise require area / (w*h) to be at least this.
    float minFillRatio = 0.0f;

    // Expected value of a normal (non-defect) pixel after normalization.
    // Keep this last to preserve positional aggregate initialization of the
    // original fields.
    float normalReference = 1.0f;
};

struct DetectionResult {
    cv::Mat brightResponse32f;
    cv::Mat darkResponse32f;
    cv::Mat brightMask;
    cv::Mat darkMask;
    cv::Mat defectMask;
    std::vector<Defect> defects;
};

class BrightDarkDetector {
public:
    explicit BrightDarkDetector(const DetectionParams& params = {});
    // validMask may be empty. Otherwise it must be CV_8UC1 with the same size;
    // zero pixels are excluded from detection and from the returned masks.
    // sourceImage is used by the blob detector; if empty, the normalized
    // image is converted to 8-bit grayscale.
    bool detect(const cv::Mat& normalized32f, DetectionResult& result,
                const cv::Mat& validMask = cv::Mat(),
                const cv::Mat& sourceImage = cv::Mat()) const;
private:
    void morphology(cv::Mat& mask, const cv::Mat& kernel) const;
    void extract(const cv::Mat& response32f, const cv::Mat& mask,
                 bool bright, cv::Mat& acceptedMask,
                 std::vector<Defect>& defects) const;
    void extractBlobs(const cv::Mat& sourceGray8u,
                      const cv::Mat& darkResponse32f,
                      const cv::Mat& validMask,
                      cv::Mat& acceptedMask,
                      std::vector<Defect>& defects) const;
    DetectionParams params_;
};

}  // namespace singledie
