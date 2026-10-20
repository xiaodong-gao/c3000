#include "SingleDiePipeline.h"

#include "BrightDarkDetector.h"
#include "DefectDetect.hpp"
#include "IntensityNormalizer.h"
#include "die_map.h"
#include "die_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cutdie {
namespace {

cv::Mat to_bgr(const cv::Mat& src) {
    if (src.empty()) {
        return {};
    }
    if (src.channels() == 3) {
        return src.clone();
    }
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

fs::path save_png(const cv::Mat& image, const fs::path& path) {
    fs::create_directories(path.parent_path());
    const std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 1};
    const fs::path tmp = path.parent_path() / (path.stem().string() + ".tmp.png");
    if (!cv::imwrite(tmp.string(), image, params)) {
        throw std::runtime_error("无法写入 " + tmp.string());
    }
    std::error_code ec;
    if (fs::exists(path)) {
        fs::remove(path, ec);
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::copy_file(tmp, path, fs::copy_options::overwrite_existing);
        fs::remove(tmp, ec);
    }
    return path;
}

cv::Rect circle_crop_rect(const WaferCircle& circle, int cols, int rows) {
    const int r = std::max(1, cvRound(circle.radius));
    const int cx = cvRound(circle.cx);
    const int cy = cvRound(circle.cy);
    const int x0 = std::max(0, cx - r);
    const int y0 = std::max(0, cy - r);
    const int x1 = std::min(cols, cx + r + 1);
    const int y1 = std::min(rows, cy + r + 1);
    if (x1 - x0 < 8 || y1 - y0 < 8) {
        return {};
    }
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

cv::Mat make_circle_mask(cv::Size size, cv::Point center, int radius) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    cv::circle(mask, center, std::max(1, radius), cv::Scalar(255), cv::FILLED, cv::LINE_8);
    return mask;
}

// 沿圆周找晶圆亮边，把半径收到亮环内侧，避免圆形 ROI 落到片外。
double radius_inside_rim(const cv::Mat& gray, const WaferCircle& circle) {
    const double expected = circle.radius;
    if (gray.empty() || expected < 8.0) {
        return expected;
    }
    const int n_angles = 360;
    const double r_lo = expected * 0.92;
    const double r_hi = expected * 1.03;
    std::vector<double> hits;
    hits.reserve(static_cast<size_t>(n_angles));
    for (int i = 0; i < n_angles; ++i) {
        const double ang = static_cast<double>(i) * (2.0 * CV_PI / n_angles);
        const double ca = std::cos(ang);
        const double sa = std::sin(ang);
        double best_r = 0.0;
        double best_v = 8.0;
        for (double r = r_lo; r <= r_hi; r += 1.0) {
            const int x = cvRound(circle.cx + r * ca);
            const int y = cvRound(circle.cy + r * sa);
            if (x < 0 || y < 0 || x >= gray.cols || y >= gray.rows) {
                continue;
            }
            const double v = gray.at<unsigned char>(y, x);
            if (v > best_v) {
                best_v = v;
                best_r = r;
            }
        }
        if (best_r > 0.0) {
            hits.push_back(best_r);
        }
    }

    double r = expected;
    if (hits.size() >= 72) {
        const auto mid = hits.begin() + static_cast<std::ptrdiff_t>(hits.size() / 2);
        std::nth_element(hits.begin(), mid, hits.end());
        r = *mid;
    } else {
        r = expected - 18.0;  // 约 TrimSize 0.9mm @ 50um
    }
    r -= 7.0;  // 收到亮环内侧，仍保住贴边真小白点

    const double max_r = std::min(std::min(circle.cx, circle.cy),
                                  std::min(gray.cols - 1.0 - circle.cx, gray.rows - 1.0 - circle.cy)) -
                         1.0;
    if (r > max_r) {
        r = max_r;
    }
    const double min_r = expected * 0.85;
    if (r < min_r) {
        r = min_r;
    }
    return r;
}

}  // namespace

InspectResult inspect_singledie(const Options& opt) {
    if (opt.image.empty() || !fs::exists(opt.image)) {
        throw std::runtime_error("未找到输入图像");
    }
    if (opt.xml.empty() || !fs::exists(opt.xml)) {
        throw std::runtime_error("未找到对齐配置 XML");
    }
    if (opt.resolution_um <= 0.0) {
        throw std::runtime_error("每个像素对应的 um 必须大于 0");
    }

    const DieMap xml = parse_die_map(opt.xml);
    std::cout << "  读取图像...\n";
    const cv::Mat full = cv::imread(opt.image.string(), cv::IMREAD_GRAYSCALE);
    if (full.empty()) {
        throw std::runtime_error("无法读取图像: " + opt.image.string());
    }

    const int lut = detect_lut_width(full);
    WaferCircle circle = make_wafer_circle(xml, full.cols, full.rows, opt.resolution_um, lut);
    const double xml_radius = circle.radius;
    circle.radius = radius_inside_rim(full, circle);
    const cv::Rect crop = circle_crop_rect(circle, full.cols, full.rows);
    if (crop.empty()) {
        throw std::runtime_error("圆形 ROI 超出图像范围");
    }

    const cv::Mat square = full(crop).clone();
    const cv::Point local_center(cvRound(circle.cx) - crop.x, cvRound(circle.cy) - crop.y);
    cv::Mat circle_mask = make_circle_mask(square.size(), local_center, cvRound(circle.radius));
    cv::Mat roi = cv::Mat::zeros(square.size(), square.type());
    square.copyTo(roi, circle_mask);
    cv::Mat gray_mask;
    cv::inRange(square, xml.gray_min, xml.gray_max, gray_mask);
    cv::Mat background_mask;
    cv::bitwise_and(circle_mask, gray_mask, background_mask);
    if (cv::countNonZero(background_mask) < 64) {
        throw std::runtime_error("圆形 ROI 有效像素过少");
    }

    fs::create_directories(opt.out);
    InspectResult result;
    result.csv_path = opt.out / "defects.csv";
    result.overlay_path = opt.out / "defect_overlay.png";

    std::cout << "  SingleDie: 圆形 ROI 圆心=(";
    {
        const auto flags = std::cout.flags();
        const auto prec = std::cout.precision();
        std::cout << std::fixed << std::setprecision(1) << circle.cx << ", " << circle.cy
                  << ")  XML半径=" << xml_radius << "px  贴边半径=" << circle.radius << "px";
        std::cout.flags(flags);
        std::cout.precision(prec);
    }
    std::cout << "  LUT=" << lut << "px\n";
    std::cout << "  截取区域: " << crop.x << "," << crop.y << " "
              << crop.width << "x" << crop.height
              << "  半径=" << (circle.radius * opt.resolution_um / 1000.0) << "mm"
              << "  灰度范围=[" << xml.gray_min << "," << xml.gray_max << "]\n";

    save_png(roi, opt.out / "wafer_circle_roi.png");
    save_png(circle_mask, opt.out / "wafer_circle_mask.png");

    singledie::IntensityNormalizer normalizer(opt.singledie_normalize);
    cv::Mat background32f;
    cv::Mat normalized32f;
    float normal_reference = 1.0f;
    if (!normalizer.process(roi, background32f, normalized32f, normal_reference, background_mask)) {
        throw std::runtime_error("SingleDie 强度归一化失败");
    }

    singledie::DetectionParams detect_params = opt.singledie_detect;
    detect_params.normalReference = normal_reference;
    singledie::BrightDarkDetector detector(detect_params);
    singledie::DetectionResult det;
    if (!detector.detect(normalized32f, det, circle_mask, roi)) {
        throw std::runtime_error("SingleDie 亮暗检出失败");
    }

    // 大白块比背景核大时会被归一化吃掉，必须在原灰度上按绝对亮度再抓一次。
    // 孤立 1x1 也会变成自己的背景，残差≈0；圆掩膜还会把亮边截成假 1x1。
    int large_white_n = 0;
    if (opt.defect.largeWhite.enable) {
        const LargeWhiteParams& wp = opt.defect.largeWhite;
        auto overlaps = [](const cv::Rect& a, const cv::Rect& b) {
            const cv::Rect hit = a & b;
            if (hit.empty()) {
                return false;
            }
            const int need = std::max(1, static_cast<int>(0.2 * std::min(a.area(), b.area())));
            return hit.area() >= need;
        };
        auto add_white = [&](const DefectCandidate& w, bool replace_overlap) {
            if (replace_overlap) {
                det.defects.erase(std::remove_if(det.defects.begin(), det.defects.end(),
                                                 [&](const singledie::Defect& d) {
                                                     return overlaps(d.bbox, w.bbox);
                                                 }),
                                  det.defects.end());
            } else {
                for (const auto& e : det.defects) {
                    if (e.bbox.contains(cv::Point(cvRound(w.centroid.x), cvRound(w.centroid.y)))) {
                        return;
                    }
                }
            }
            singledie::Defect d;
            d.id = static_cast<int>(det.defects.size());
            d.bright = true;
            d.bbox = w.bbox;
            d.area = w.area;
            d.center = w.centroid;
            d.meanResponse = w.meanL2 / 255.0;
            d.maxResponse = w.maxL2 / 255.0;
            det.defects.push_back(d);
            ++large_white_n;
        };

        const int inner_m = std::max(0, wp.isolatedInnerMargin);
        const int near_keep = std::max(0, wp.isolatedNearKeep);
        const int far_m = std::max(0, wp.isolatedFarMargin);
        auto rim_of = [&](const cv::Point2d& c) {
            const double dist = std::hypot(c.x - local_center.x, c.y - local_center.y);
            return circle.radius - dist;
        };
        auto near_rim = [&](const cv::Point2d& c) {
            return rim_of(c) <= static_cast<double>(inner_m);
        };

        std::vector<DefectCandidate> whites = extractLargeWhiteRegions(roi, wp);
        for (const auto& w : whites) {
            const double rim = rim_of(w.centroid);
            if (near_keep > 0) {
                // 1px 近/远分档：inner_m 只作用在面积 1，避免误删贴边 2px 真点。
                if (w.area <= 1) {
                    if (inner_m > 0 && rim <= static_cast<double>(inner_m)) {
                        continue;
                    }
                    const bool keep_near = rim <= static_cast<double>(near_keep);
                    const bool keep_far = far_m > 0 && rim >= static_cast<double>(far_m);
                    if (!keep_near && !keep_far) {
                        continue;
                    }
                }
            } else {
                if (inner_m > 0 && w.area <= 4 && near_rim(w.centroid)) {
                    continue;
                }
                // nearKeep=0 时 farMargin 当作 1px 距边上限：丢掉片内深处 1px，保住顶部近边真点。
                if (far_m > 0 && w.area <= 1 && rim >= static_cast<double>(far_m)) {
                    continue;
                }
            }
            add_white(w, true);
        }

        if (wp.isolatedMaxArea > 0 && wp.isolatedThreshold > 0.0) {
            // 在未抠圆的原图上连通：圆掩膜会把亮边截成假 1x1，内缩又会丢掉底部贴边真白点。
            auto peak_covered = [&](const cv::Point& p) {
                for (const auto& e : det.defects) {
                    if (e.bbox.contains(p)) {
                        return true;
                    }
                }
                return false;
            };

            cv::Mat tiny;
            cv::compare(square, wp.isolatedThreshold, tiny, cv::CMP_GE);
            cv::Mat labels, stats, centroids;
            const int ncc =
                cv::connectedComponentsWithStats(tiny, labels, stats, centroids, 8, CV_32S);
            const int max_side =
                std::max(2, cvRound(std::sqrt(static_cast<double>(wp.isolatedMaxArea)) + 2.0));
            for (int i = 1; i < ncc; ++i) {
                const int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area < 1 || area > wp.isolatedMaxArea) {
                    continue;
                }
                DefectCandidate c;
                c.area = area;
                c.bbox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT),
                                  stats.at<int>(i, cv::CC_STAT_TOP),
                                  stats.at<int>(i, cv::CC_STAT_WIDTH),
                                  stats.at<int>(i, cv::CC_STAT_HEIGHT));
                if (c.bbox.width > max_side || c.bbox.height > max_side) {
                    continue;
                }
                c.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
                const int ix = cvRound(c.centroid.x);
                const int iy = cvRound(c.centroid.y);
                if (ix < 0 || iy < 0 || ix >= circle_mask.cols || iy >= circle_mask.rows ||
                    circle_mask.at<unsigned char>(iy, ix) == 0) {
                    continue;
                }
                double sum = 0.0;
                double peak = -1.0;
                int n = 0;
                cv::Point peak_pt(c.bbox.x, c.bbox.y);
                const cv::Rect box = c.bbox & cv::Rect(0, 0, square.cols, square.rows);
                for (int y = box.y; y < box.y + box.height; ++y) {
                    const int* lab = labels.ptr<int>(y);
                    const unsigned char* row = square.ptr<unsigned char>(y);
                    for (int x = box.x; x < box.x + box.width; ++x) {
                        if (lab[x] != i) {
                            continue;
                        }
                        const double v = static_cast<double>(row[x]);
                        sum += v;
                        if (v >= peak) {
                            peak = v;
                            peak_pt = cv::Point(x, y);
                        }
                        ++n;
                    }
                }
                if (n > 0) {
                    c.meanL2 = sum / static_cast<double>(n);
                    c.maxL2 = peak;
                }
                if (peak_covered(peak_pt)) {
                    continue;
                }
                add_white(c, false);
            }

            // 贴边小白点若和亮环连在一起，连通域会超面积上限；再收圆内未被框住的局部峰值。
            const int th = std::max(1, cvRound(wp.isolatedThreshold));
            for (int y = 1; y < square.rows - 1; ++y) {
                const unsigned char* row = square.ptr<unsigned char>(y);
                const unsigned char* up = square.ptr<unsigned char>(y - 1);
                const unsigned char* dn = square.ptr<unsigned char>(y + 1);
                const unsigned char* mask = circle_mask.ptr<unsigned char>(y);
                for (int x = 1; x < square.cols - 1; ++x) {
                    if (mask[x] == 0) {
                        continue;
                    }
                    const int v = static_cast<int>(row[x]);
                    if (v < th) {
                        continue;
                    }
                    const cv::Point p(x, y);
                    if (peak_covered(p)) {
                        continue;
                    }
                    const int nmax = std::max({static_cast<int>(up[x - 1]), static_cast<int>(up[x]),
                                               static_cast<int>(up[x + 1]), static_cast<int>(row[x - 1]),
                                               static_cast<int>(row[x + 1]), static_cast<int>(dn[x - 1]),
                                               static_cast<int>(dn[x]), static_cast<int>(dn[x + 1])});
                    if (nmax >= v) {
                        continue;
                    }
                    DefectCandidate c;
                    c.area = 1;
                    c.bbox = cv::Rect(x, y, 1, 1);
                    c.centroid = cv::Point2d(x, y);
                    c.meanL2 = c.maxL2 = static_cast<double>(v);
                    add_white(c, false);
                }
            }
        }
    }

    cv::Mat overlay = to_bgr(full);
    const int preview_px = opt.overlay_preview_px > 0 ? opt.overlay_preview_px : kDetectTarget;
    const double scale = std::max(overlay.cols, overlay.rows) / static_cast<double>(preview_px);
    const int thickness = std::max(1, cvRound(scale));
    cv::circle(overlay, cv::Point(cvRound(circle.cx), cvRound(circle.cy)), cvRound(circle.radius),
               cv::Scalar(255, 180, 0), thickness, cv::LINE_AA);

    const cv::Rect roi_bounds(0, 0, roi.cols, roi.rows);
    const cv::Rect full_bounds(0, 0, overlay.cols, overlay.rows);
    int rec_id = 0;
    for (const auto& src : det.defects) {
        WaferDefect d;
        d.die_x = 0;
        d.die_y = 0;
        d.id = ++rec_id;
        d.bbox = src.bbox & roi_bounds;
        d.centroid = src.center;
        d.area = src.area;
        d.meanL2 = src.meanResponse;
        d.maxL2 = src.maxResponse;
        d.meanSigned = src.bright ? src.meanResponse : -src.meanResponse;
        d.brightArea = src.bright ? src.area : 0;
        d.darkArea = src.bright ? 0 : src.area;
        d.polarity = src.bright ? DefectPolarity::Bright : DefectPolarity::Dark;
        if (d.bbox.width <= 0 || d.bbox.height <= 0) {
            continue;
        }
        result.defects.push_back(d);

        const cv::Scalar color = src.bright ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 80, 0);
        cv::Rect vis(d.bbox.x + crop.x, d.bbox.y + crop.y, d.bbox.width, d.bbox.height);
        const int min_vis = 16;
        if (vis.width < min_vis || vis.height < min_vis) {
            const int cx = vis.x + vis.width / 2;
            const int cy = vis.y + vis.height / 2;
            vis.x = cx - min_vis / 2;
            vis.y = cy - min_vis / 2;
            vis.width = min_vis;
            vis.height = min_vis;
            vis &= full_bounds;
        }
        cv::rectangle(overlay, vis.area() > 0 ? vis : cv::Rect(d.bbox.x + crop.x, d.bbox.y + crop.y,
                                                              d.bbox.width, d.bbox.height),
                      color, std::max(1, thickness / 2));
        const cv::Point center(cvRound(d.centroid.x) + crop.x, cvRound(d.centroid.y) + crop.y);
        if (full_bounds.contains(center)) {
            cv::circle(overlay, center, std::max(1, thickness), color, cv::FILLED, cv::LINE_AA);
        }
    }

    result.inspected = 1;
    result.defect_count = static_cast<int>(result.defects.size());
    result.overlay_path = save_png(overlay, result.overlay_path);

    result.die_overlay_dir = opt.out / "defects";
    fs::create_directories(result.die_overlay_dir);
    const cv::Mat roi_bgr = to_bgr(roi);
    for (const auto& d : result.defects) {
        const int pad = std::max(32, std::max(d.bbox.width, d.bbox.height));
        cv::Rect crop_box(d.bbox.x - pad, d.bbox.y - pad, d.bbox.width + 2 * pad,
                          d.bbox.height + 2 * pad);
        crop_box &= roi_bounds;
        if (crop_box.empty()) {
            continue;
        }
        cv::Mat view = roi_bgr(crop_box).clone();
        const cv::Rect view_bounds(0, 0, view.cols, view.rows);
        for (const auto& other : result.defects) {
            cv::Rect local(other.bbox.x - crop_box.x, other.bbox.y - crop_box.y, other.bbox.width,
                           other.bbox.height);
            local &= view_bounds;
            if (local.area() > 0) {
                cv::rectangle(view, local, cv::Scalar(0, 0, 255), 2);
            }
        }
        std::ostringstream name;
        name << "defect_" << std::setfill('0') << std::setw(2) << d.id << ".png";
        save_png(view, result.die_overlay_dir / name.str());
    }

    std::ofstream csv(result.csv_path, std::ios::binary);
    if (!csv) {
        throw std::runtime_error("无法写入 " + result.csv_path.string());
    }
    csv << "origin_x,origin_y,id,polarity,x,y,w,h,area,cx,cy,meanL2,maxL2,meanSigned,brightArea,darkArea\r\n";
    for (const auto& d : result.defects) {
        csv << crop.x << ',' << crop.y << ',' << d.id << ',' << defectPolarityName(d.polarity)
            << ',' << d.bbox.x << ',' << d.bbox.y << ',' << d.bbox.width << ',' << d.bbox.height
            << ',' << d.area << ',' << d.centroid.x << ',' << d.centroid.y << ',' << d.meanL2 << ','
            << d.maxL2 << ',' << d.meanSigned << ',' << d.brightArea << ',' << d.darkArea
            << "\r\n";
    }

    std::cout << "  SingleDie 完成: 缺陷 " << result.defect_count
              << "  (大白块 " << large_white_n << ")\n";
    std::cout << "  圆形 ROI: " << (opt.out / "wafer_circle_roi.png").string() << '\n';
    std::cout << "  缺陷原图: " << result.overlay_path.string()
              << "  (画在原始图像上)\n";
    std::cout << "  单点裁剪: " << result.die_overlay_dir.string() << '\n';
    std::cout << "  缺陷清单: " << result.csv_path.string()
              << "  (坐标相对圆形 ROI，原图偏移 " << crop.x << "," << crop.y << ")\n";
    return result;
}

}  // namespace cutdie
