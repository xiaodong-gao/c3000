#include "DieToDatabase.h"

#include "Die2DatabasePipeline.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cutdie {
namespace {

cv::Mat crop_die(const cv::Mat& full, const DieBox& box, cv::Rect& crop) {
    const int left = std::max(box.left, 0);
    const int top = std::max(box.top, 0);
    const int right = std::min(box.right, full.cols);
    const int bottom = std::min(box.bottom, full.rows);
    if (right - left < 8 || bottom - top < 8) {
        crop = {};
        return {};
    }
    crop = cv::Rect(left, top, right - left, bottom - top);
    return full(crop).clone();
}

bool match_pair_size(cv::Mat& templ, cv::Mat& data) {
    const int w = std::min(templ.cols, data.cols);
    const int h = std::min(templ.rows, data.rows);
    if (w < 16 || h < 16) {
        return false;
    }
    if (templ.cols != w || templ.rows != h) {
        templ = templ(cv::Rect(0, 0, w, h)).clone();
    }
    if (data.cols != w || data.rows != h) {
        data = data(cv::Rect(0, 0, w, h)).clone();
    }
    return true;
}

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

cv::Mat load_reference_patch(const fs::path& path) {
    if (path.empty()) {
        throw std::runtime_error("DieToDatabase 未指定参考 patch 模板");
    }
    if (!fs::exists(path)) {
        throw std::runtime_error("参考 patch 模板不存在: " + path.string());
    }
    if (fs::is_directory(path)) {
        throw std::runtime_error("DieToDatabase 需要一张参考 patch 图像，不是文件夹: " + path.string());
    }
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("无法读取参考 patch 模板: " + path.string());
    }
    return image;
}

void append_unique(std::vector<DefectCandidate>& dst, const std::vector<DefectCandidate>& extra,
                   DefectPolarity polarity) {
    for (const auto& item : extra) {
        bool overlap = false;
        for (const auto& kept : dst) {
            if (kept.polarity != polarity) {
                continue;
            }
            const cv::Rect hit = kept.bbox & item.bbox;
            const int need = std::max(1, static_cast<int>(0.3 * item.bbox.area()));
            if (hit.area() >= need) {
                overlap = true;
                break;
            }
        }
        if (!overlap) {
            dst.push_back(item);
        }
    }
}

const char* polarity_name(DefectPolarity polarity) {
    switch (polarity) {
        case DefectPolarity::Bright:
            return "bright";
        case DefectPolarity::Dark:
            return "dark";
        case DefectPolarity::Mixed:
            return "mixed";
        default:
            return "unknown";
    }
}

std::vector<DefectCandidate> candidates_on_patch(const DatabaseMatch& matched, const cv::Mat& data,
                                                 DatabaseComparator& comparator, const Options& opt) {
    std::vector<DefectCandidate> found;
    found.reserve(matched.candidates.size() + 8);
    for (DefectCandidate candidate : matched.candidates) {
        candidate.bbox.x += matched.commonRoi.x;
        candidate.bbox.y += matched.commonRoi.y;
        candidate.centroid.x += matched.commonRoi.x;
        candidate.centroid.y += matched.commonRoi.y;
        found.push_back(candidate);
    }
    if (opt.defect.largeWhite.enable) {
        append_unique(found, comparator.findBrightBlocks(data), DefectPolarity::Bright);
    }
    if (opt.defect.darkRegion.enable) {
        append_unique(found, comparator.findDarkBlocks(data), DefectPolarity::Dark);
    }
    // 已经有大白块的晶粒只保留白块。旁边的差分暗点是白块把亮度归一带偏造成的。
    const bool hasLargeWhite = std::any_of(found.begin(), found.end(), [](const DefectCandidate& candidate) {
        return candidate.polarity == DefectPolarity::Bright && candidate.area >= 1000;
    });
    if (hasLargeWhite) {
        found.erase(std::remove_if(found.begin(), found.end(),
                                    [](const DefectCandidate& candidate) {
                                        return candidate.polarity != DefectPolarity::Bright;
                                    }),
                    found.end());
    }
    int id = 0;
    for (auto& candidate : found) {
        candidate.id = ++id;
    }
    return found;
}

}  // namespace

InspectResult inspect_die_to_database(const Options& opt, const CutResult& cut) {
    if (cut.full.empty()) {
        throw std::runtime_error("切割结果中没有原始晶圆图");
    }
    const cv::Mat reference = load_reference_patch(opt.template_path);
    DatabaseComparator comparator(opt);

    cv::Mat overlay = to_bgr(cut.full);
    const int preview_px = opt.overlay_preview_px > 0 ? opt.overlay_preview_px : kDetectTarget;
    const double scale = std::max(overlay.cols, overlay.rows) / static_cast<double>(preview_px);
    const int thickness = std::max(1, cvRound(scale));
    const cv::Rect wafer_bounds(0, 0, overlay.cols, overlay.rows);

    InspectResult result;
    result.csv_path = opt.out / "defects.csv";
    result.overlay_path = opt.out / "defect_overlay.png";
    fs::create_directories(opt.out);

    std::cout << "  参考 patch: " << opt.template_path.string() << "  " << reference.cols << "x"
              << reference.rows << '\n';
    std::cout << "  检出后端: " << comparator.backendText() << '\n';

    std::vector<std::vector<cv::Point>> all_contours;
    for (const auto& box : cut.boxes) {
        if (box.skipped && !opt.keep_skipped) {
            ++result.skipped_die;
            continue;
        }
        if (opt.skip_inspect.count({box.x_idx, box.y_idx}) > 0) {
            ++result.skipped_die;
            continue;
        }
        cv::Rect crop;
        cv::Mat data = crop_die(cut.full, box, crop);
        if (data.empty()) {
            ++result.skipped_die;
            continue;
        }

        cv::Mat templ = reference.clone();
        cv::Mat data_one = data.clone();
        if (!match_pair_size(templ, data_one)) {
            std::cerr << "    跳过 die_x" << box.x_idx << "_y" << box.y_idx << ": 与参考 patch 重叠过小\n";
            ++result.skipped_no_template;
            cv::rectangle(overlay, crop, cv::Scalar(80, 80, 255), thickness);
            continue;
        }

        const DatabaseMatch matched = comparator.match(templ, data_one);
        if (!matched.ok) {
            std::cerr << "    失败 die_x" << box.x_idx << "_y" << box.y_idx << ": " << matched.error << '\n';
            ++result.failed;
            cv::rectangle(overlay, crop, cv::Scalar(0, 165, 255), thickness);
            continue;
        }

        const std::vector<DefectCandidate> found = candidates_on_patch(matched, data, comparator, opt);
        ++result.inspected;
        if (result.inspected % 50 == 0) {
            std::cout << "    对比进度 " << result.inspected << "/" << cut.boxes.size() << "  缺陷 "
                      << result.defects.size() << '\n';
        }

        const cv::Scalar die_color = found.empty() ? cv::Scalar(90, 255, 0) : cv::Scalar(0, 200, 255);
        cv::rectangle(overlay, crop, die_color, thickness);
        if (!found.empty()) {
            if (result.die_overlay_dir.empty()) {
                result.die_overlay_dir = opt.out / "die_defects";
                fs::create_directories(result.die_overlay_dir);
            }
            save_png(comparator.draw(data, found),
                     result.die_overlay_dir / box.file_name(opt.image.stem().string()));
        }

        int min_vis = std::max(crop.width, crop.height) / 4;
        if (min_vis < 6) {
            min_vis = 6;
        }
        if (min_vis > 24) {
            min_vis = 24;
        }
        for (const DefectCandidate& candidate : found) {
            WaferDefect defect;
            defect.die_x = box.x_idx;
            defect.die_y = box.y_idx;
            defect.id = candidate.id;
            defect.bbox = cv::Rect(crop.x + candidate.bbox.x, crop.y + candidate.bbox.y, candidate.bbox.width,
                                   candidate.bbox.height);
            defect.bbox &= wafer_bounds;
            defect.centroid = cv::Point2d(crop.x + candidate.centroid.x, crop.y + candidate.centroid.y);
            defect.area = candidate.area;
            defect.meanL2 = candidate.meanL2;
            defect.maxL2 = candidate.maxL2;
            defect.meanSigned = candidate.meanSigned;
            defect.brightArea = candidate.brightArea;
            defect.darkArea = candidate.darkArea;
            defect.polarity = candidate.polarity;
            if (defect.bbox.width <= 0 || defect.bbox.height <= 0) {
                continue;
            }
            result.defects.push_back(defect);

            cv::Scalar color(0, 0, 255);
            if (candidate.polarity == DefectPolarity::Dark) {
                color = cv::Scalar(255, 80, 0);
            } else if (candidate.polarity == DefectPolarity::Mixed) {
                color = cv::Scalar(0, 200, 255);
            } else if (candidate.polarity == DefectPolarity::Unknown) {
                color = cv::Scalar(180, 180, 180);
            }
            cv::Rect vis = defect.bbox;
            if (vis.width < min_vis || vis.height < min_vis) {
                const int cx = vis.x + vis.width / 2;
                const int cy = vis.y + vis.height / 2;
                vis.x = cx - min_vis / 2;
                vis.y = cy - min_vis / 2;
                vis.width = min_vis;
                vis.height = min_vis;
                vis &= wafer_bounds;
            }
            cv::rectangle(overlay, vis.area() > 0 ? vis : defect.bbox, color, 1);
            const cv::Point center(cvRound(defect.centroid.x), cvRound(defect.centroid.y));
            if (wafer_bounds.contains(center)) {
                cv::circle(overlay, center, 1, color, cv::FILLED, cv::LINE_AA);
            }
        }

        if (!matched.blobMask.empty()) {
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(matched.blobMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (auto& contour : contours) {
                for (auto& pt : contour) {
                    pt.x += crop.x + matched.commonRoi.x;
                    pt.y += crop.y + matched.commonRoi.y;
                }
                all_contours.push_back(std::move(contour));
            }
        }
    }

    if (!all_contours.empty()) {
        cv::drawContours(overlay, all_contours, -1, cv::Scalar(0, 0, 255), std::max(1, thickness / 2));
    }
    result.defect_count = static_cast<int>(result.defects.size());
    result.overlay_path = save_png(overlay, result.overlay_path);

    std::ofstream csv(result.csv_path, std::ios::binary);
    if (!csv) {
        throw std::runtime_error("无法写入 " + result.csv_path.string());
    }
    csv << "die_x,die_y,id,polarity,x,y,w,h,area,cx,cy,meanL2,maxL2,meanSigned,brightArea,darkArea\r\n";
    for (const auto& defect : result.defects) {
        csv << defect.die_x << ',' << defect.die_y << ',' << defect.id << ',' << polarity_name(defect.polarity)
            << ',' << defect.bbox.x << ',' << defect.bbox.y << ',' << defect.bbox.width << ','
            << defect.bbox.height << ',' << defect.area << ',' << defect.centroid.x << ',' << defect.centroid.y
            << ',' << defect.meanL2 << ',' << defect.maxL2 << ',' << defect.meanSigned << ','
            << defect.brightArea << ',' << defect.darkArea << "\r\n";
    }

    std::cout << "  对比完成: 成功 " << result.inspected << "  失败 " << result.failed << "  无重叠 "
              << result.skipped_no_template << "  跳过 " << result.skipped_die << "  缺陷 "
              << result.defect_count << '\n';
    std::cout << "  缺陷原图: " << result.overlay_path.string() << '\n';
    std::cout << "  单颗 die 标注: " << result.die_overlay_dir.string() << '\n';
    std::cout << "  缺陷清单: " << result.csv_path.string() << '\n';
    return result;
}

}  // namespace cutdie
