#include "Mssim.hpp"
#include "Mssim.cuh"

#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

MssimResult computeMssimSingleChannel(const cv::Mat& image1, const cv::Mat& image2,
                                      const cv::Mat& mask) {
    CV_Assert(image1.size() == image2.size());
    CV_Assert(image1.type() == CV_8UC1 && image2.type() == CV_8UC1);

    cv::Mat f1;
    cv::Mat f2;
    image1.convertTo(f1, CV_32F);
    image2.convertTo(f2, CV_32F);

    constexpr double k1 = 0.01;
    constexpr double k2 = 0.03;
    constexpr double L = 255.0;
    const float C1 = static_cast<float>((k1 * L) * (k1 * L));
    const float C2 = static_cast<float>((k2 * L) * (k2 * L));

    cv::Mat mu1;
    cv::Mat mu2;
    cv::GaussianBlur(f1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(f2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1Sq = mu1.mul(mu1);
    cv::Mat mu2Sq = mu2.mul(mu2);
    cv::Mat mu1Mu2 = mu1.mul(mu2);

    cv::Mat sigma1Sq;
    cv::Mat sigma2Sq;
    cv::Mat sigma12;
    cv::GaussianBlur(f1.mul(f1), sigma1Sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(f2.mul(f2), sigma2Sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(f1.mul(f2), sigma12, cv::Size(11, 11), 1.5);
    sigma1Sq -= mu1Sq;
    sigma2Sq -= mu2Sq;
    sigma12 -= mu1Mu2;

    cv::Mat numerator;
    cv::Mat denominator;
    cv::multiply(2.0f * mu1Mu2 + C1, 2.0f * sigma12 + C2, numerator);
    cv::multiply(mu1Sq + mu2Sq + C1, sigma1Sq + sigma2Sq + C2, denominator);

    MssimResult result;
    cv::divide(numerator, denominator, result.map);

    if (!mask.empty()) {
        cv::Mat eroded;
        cv::erode(mask, eroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
        result.mssim = cv::mean(result.map, eroded)[0];
    } else {
        const int border = 5;
        const cv::Rect inner(border, border, result.map.cols - 2 * border, result.map.rows - 2 * border);
        result.mssim = cv::mean(result.map(inner))[0];
    }
    return result;
}

}  // namespace

MssimResult computeMssim(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& mask) {
    CV_Assert(image1.size() == image2.size());
    CV_Assert(image1.type() == image2.type());
    CV_Assert(image1.type() == CV_8UC1 || image1.type() == CV_8UC3);

    if (image1.channels() == 1) {
        return computeMssimSingleChannel(image1, image2, mask);
    }

    std::vector<cv::Mat> ch1;
    std::vector<cv::Mat> ch2;
    cv::split(image1, ch1);
    cv::split(image2, ch2);

    cv::Mat acc;
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        const MssimResult channel = computeMssimSingleChannel(ch1[i], ch2[i], mask);
        if (acc.empty()) {
            acc = channel.map.clone();
        } else {
            acc += channel.map;
        }
        sum += channel.mssim;
    }

    MssimResult result;
    result.map = acc / 3.0;
    result.mssim = sum / 3.0;
    return result;
}

MssimResult computeMssimCuda(MssimCuda& engine, const cv::Mat& image1, const cv::Mat& image2) {
    CV_Assert(image1.size() == image2.size());
    CV_Assert(image1.type() == image2.type());
    CV_Assert(image1.type() == CV_8UC1 || image1.type() == CV_8UC3);

    const cv::Mat a = image1.isContinuous() ? image1 : image1.clone();
    const cv::Mat b = image2.isContinuous() ? image2 : image2.clone();

    MssimResult result;
    result.map.create(a.rows, a.cols, CV_32F);
    const cudaError_t status =
        engine.compute(a.data, b.data, a.cols, a.rows, a.channels(),
                       result.map.ptr<float>(), result.mssim);
    if (status != cudaSuccess) {
        throw cv::Exception(cv::Error::StsError, cudaGetErrorString(status),
                            "computeMssimCuda", __FILE__, __LINE__);
    }
    return result;
}

cv::Mat fuseByMssimCuda(MssimCuda& engine, const cv::Mat& templateImg, const cv::Mat& alignedImg,
                        const cv::Mat& ssimMap) {
    CV_Assert(templateImg.size() == alignedImg.size());
    CV_Assert(templateImg.type() == alignedImg.type());
    CV_Assert(ssimMap.size() == templateImg.size());
    CV_Assert(ssimMap.type() == CV_32F);

    const cv::Mat a = templateImg.isContinuous() ? templateImg : templateImg.clone();
    const cv::Mat b = alignedImg.isContinuous() ? alignedImg : alignedImg.clone();
    const cv::Mat w = ssimMap.isContinuous() ? ssimMap : ssimMap.clone();

    cv::Mat fused(a.size(), a.type());
    const cudaError_t status =
        engine.fuse(a.data, b.data, w.ptr<float>(), a.cols, a.rows, a.channels(), fused.data);
    if (status != cudaSuccess) {
        throw cv::Exception(cv::Error::StsError, cudaGetErrorString(status), "fuseByMssimCuda",
                            __FILE__, __LINE__);
    }
    return fused;
}

cv::Mat fuseByMssim(const cv::Mat& templateImg, const cv::Mat& alignedImg, const cv::Mat& ssimMap) {
    CV_Assert(templateImg.size() == alignedImg.size());
    CV_Assert(templateImg.type() == alignedImg.type());
    CV_Assert(ssimMap.size() == templateImg.size());
    CV_Assert(ssimMap.type() == CV_32F);

    cv::Mat templateF;
    cv::Mat alignedF;
    templateImg.convertTo(templateF, CV_32F);
    alignedImg.convertTo(alignedF, CV_32F);

    cv::Mat weight;
    if (templateImg.channels() == 1) {
        weight = ssimMap;
    } else {
        cv::merge(std::vector<cv::Mat>{ssimMap, ssimMap, ssimMap}, weight);
    }

    cv::Mat fusedF = templateF.mul(weight) + alignedF.mul(cv::Scalar::all(1.0) - weight);
    cv::Mat fused;
    fusedF.convertTo(fused, templateImg.type());
    return fused;
}

cv::Mat ssimMapToGray(const cv::Mat& ssimMap) {
    cv::Mat gray;
    ssimMap.convertTo(gray, CV_8U, 255.0);
    return gray;
}

cv::Mat ssimMapToColor(const cv::Mat& ssimMap) {
    cv::Mat color;
    cv::applyColorMap(ssimMapToGray(ssimMap), color, cv::COLORMAP_JET);
    return color;
}
