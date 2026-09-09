#include "PhaseCorrelate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

cv::Point2d subpixelPeak(const cv::Mat& corr, const cv::Point& peak) {
    const int w = corr.cols;
    const int h = corr.rows;
    double sum = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int x = (peak.x + dx + w) % w;
            const int y = (peak.y + dy + h) % h;
            const double v = std::max(0.0, static_cast<double>(corr.at<float>(y, x)));
            sum += v;
            sx += v * static_cast<double>(peak.x + dx);
            sy += v * static_cast<double>(peak.y + dy);
        }
    }
    if (sum <= 1e-12) {
        return cv::Point2d(peak.x, peak.y);
    }
    return cv::Point2d(sx / sum, sy / sum);
}

cv::Point2d wrapShift(cv::Point2d shift, int width, int height) {
    if (shift.x > width * 0.5) {
        shift.x -= width;
    }
    if (shift.y > height * 0.5) {
        shift.y -= height;
    }
    return shift;
}

}  // namespace

cv::Point2d phaseShiftBinary(const cv::Mat& templateBin, const cv::Mat& dataBin, double* response) {
    CV_Assert(templateBin.size() == dataBin.size());
    CV_Assert(templateBin.type() == CV_8UC1 && dataBin.type() == CV_8UC1);

    cv::Mat templateF;
    cv::Mat dataF;
    templateBin.convertTo(templateF, CV_32F);
    dataBin.convertTo(dataF, CV_32F);

    cv::Mat hanning;
    cv::createHanningWindow(hanning, templateF.size(), CV_32F);
    cv::multiply(templateF, hanning, templateF);
    cv::multiply(dataF, hanning, dataF);

    cv::Mat fft1;
    cv::Mat fft2;
    cv::dft(templateF, fft1, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(dataF, fft2, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat cross;
    cv::mulSpectrums(fft1, fft2, cross, 0, true);

    std::vector<cv::Mat> planes(2);
    cv::split(cross, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);
    mag += 1e-12f;
    planes[0] /= mag;
    planes[1] /= mag;
    cv::merge(planes, cross);

    cv::Mat corr;
    cv::idft(cross, corr, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    corr.convertTo(corr, CV_32F);

    const int w = corr.cols;
    const int h = corr.rows;
    const int maxShift =
        std::max(1, std::min(std::min(w, h), std::max(16, std::min(48, std::min(w, h) / 8))));

    struct Candidate {
        cv::Point loc;
        double value;
        cv::Point2d shift;
    };

    auto makeCandidate = [&](const cv::Rect& roi) {
        cv::Mat region = corr(roi);
        double minVal = 0.0;
        double maxVal = 0.0;
        cv::Point minLoc;
        cv::Point maxLoc;
        cv::minMaxLoc(region, &minVal, &maxVal, &minLoc, &maxLoc);
        Candidate candidate;
        candidate.loc = cv::Point(roi.x + maxLoc.x, roi.y + maxLoc.y);
        candidate.value = maxVal;
        candidate.shift = wrapShift(subpixelPeak(corr, candidate.loc), w, h);
        candidate.shift.x = -candidate.shift.x;
        candidate.shift.y = -candidate.shift.y;
        return candidate;
    };

    Candidate candidates[] = {
        makeCandidate(cv::Rect(0, 0, maxShift, maxShift)),
        makeCandidate(cv::Rect(w - maxShift, 0, maxShift, maxShift)),
        makeCandidate(cv::Rect(0, h - maxShift, maxShift, maxShift)),
        makeCandidate(cv::Rect(w - maxShift, h - maxShift, maxShift, maxShift)),
    };

    const Candidate* best = &candidates[0];
    for (const Candidate& candidate : candidates) {
        const double bestMag = std::hypot(best->shift.x, best->shift.y);
        const double mag = std::hypot(candidate.shift.x, candidate.shift.y);
        if (candidate.value > best->value + 1e-12 ||
            (std::abs(candidate.value - best->value) <= 1e-12 && mag < bestMag)) {
            best = &candidate;
        }
    }

    if (response) {
        *response = best->value;
    }
    return best->shift;
}

cv::Rect commonOverlapRoi(cv::Size size, const cv::Point2d& shift) {
    const double w = static_cast<double>(size.width);
    const double h = static_cast<double>(size.height);
    const int x0 = static_cast<int>(std::ceil(std::max(0.0, -shift.x)));
    const int y0 = static_cast<int>(std::ceil(std::max(0.0, -shift.y)));
    const int x1 = static_cast<int>(std::floor(std::min(w, w - shift.x)));
    const int y1 = static_cast<int>(std::floor(std::min(h, h - shift.y)));
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

cv::Mat shiftImage(const cv::Mat& src, const cv::Point2d& shift, int interpolation) {
    cv::Mat affine = cv::Mat::zeros(2, 3, CV_64F);
    affine.at<double>(0, 0) = 1.0;
    affine.at<double>(0, 2) = -shift.x;
    affine.at<double>(1, 1) = 1.0;
    affine.at<double>(1, 2) = -shift.y;
    cv::Mat dst;
    cv::warpAffine(src, dst, affine, src.size(), interpolation, cv::BORDER_CONSTANT, cv::Scalar(0));
    return dst;
}
