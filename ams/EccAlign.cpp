#include "EccAlign.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

EccAlignResult eccShift(const cv::Mat& templateImg, const cv::Mat& dataImg,
                        const EccAlignParams& params) {
    EccAlignResult result;
    if (templateImg.empty() || dataImg.empty() || templateImg.size() != dataImg.size() ||
        templateImg.type() != CV_8UC1 || dataImg.type() != CV_8UC1) {
        result.error = "ECC requires same-size CV_8UC1 images";
        return result;
    }
    if (params.gaussFiltSize < 1 || (params.gaussFiltSize % 2) == 0) {
        result.error = "ECC gaussFiltSize must be a positive odd integer";
        return result;
    }
    if (params.motionType != cv::MOTION_TRANSLATION && params.motionType != cv::MOTION_EUCLIDEAN &&
        params.motionType != cv::MOTION_AFFINE && params.motionType != cv::MOTION_HOMOGRAPHY) {
        result.error =
            "ECC motionType must be MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE or "
            "MOTION_HOMOGRAPHY";
        return result;
    }

    cv::Mat templateF;
    cv::Mat dataF;
    templateImg.convertTo(templateF, CV_32F);
    dataImg.convertTo(dataF, CV_32F);

    cv::Mat warp = params.motionType == cv::MOTION_HOMOGRAPHY ? cv::Mat::eye(3, 3, CV_32F)
                                                              : cv::Mat::eye(2, 3, CV_32F);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                    params.maxIters, params.eps);

    try {
        result.cc = cv::findTransformECC(templateF, dataF, warp, params.motionType, criteria,
                                         cv::noArray(), params.gaussFiltSize);
        // OpenCV warp is [1 0 tx; 0 1 ty] with WARP_INVERSE_MAP: dst(x)=src(x+tx).
        // shiftImage(src, s) does the same with s=(tx, ty). Report that aligning shift.
        result.shift.x = static_cast<double>(warp.at<float>(0, 2));
        result.shift.y = static_cast<double>(warp.at<float>(1, 2));
        result.ok = true;
    } catch (const cv::Exception& e) {
        result.error = e.err.empty() ? e.what() : e.err;
    }
    return result;
}
