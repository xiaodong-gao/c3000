#include "die_cut.h"

#include "die_map.h"
#include "die_util.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cutdie {
namespace {

void clear_old_pngs(const fs::path& dir) {
    if (!fs::exists(dir)) {
        return;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && is_png(entry.path())) {
            std::error_code ec;
            fs::remove(entry.path(), ec);
        }
    }
}

cv::Mat make_detect_image(const cv::Mat& full, double& scale, int preview_px) {
    const int longest = std::max(full.cols, full.rows);
    const int target = preview_px > 0 ? preview_px : kDetectTarget;
    if (longest <= target) {
        scale = 1.0;
        return full;
    }
    scale = static_cast<double>(longest) / static_cast<double>(target);
    const int new_w = std::max(1, py_int(py_round(full.cols / scale)));
    const int new_h = std::max(1, py_int(py_round(full.rows / scale)));
    cv::Mat preview;
    cv::resize(full, preview, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    return preview;
}

cv::Mat draw_overlay(
    const cv::Mat& gray,
    const std::vector<DieBox>& boxes,
    double cx,
    double cy,
    double radius
) {
    cv::Mat overlay;
    cv::cvtColor(gray, overlay, cv::COLOR_GRAY2BGR);
    cv::circle(overlay, cv::Point(cvRound(cx), cvRound(cy)), cvRound(radius),
               cv::Scalar(255, 180, 0), 1);
    for (const auto& box : boxes) {
        const cv::Scalar color = box.skipped ? cv::Scalar(80, 80, 255) : cv::Scalar(90, 255, 0);
        cv::rectangle(
            overlay,
            cv::Point(box.left, box.top),
            cv::Point(box.right - 1, box.bottom - 1),
            color,
            1);
    }
    return overlay;
}

fs::path save_overlay(const cv::Mat& overlay, const fs::path& out_dir) {
    fs::path overlay_path = out_dir / "die_overlay.png";
    const fs::path tmp_overlay = out_dir / "die_overlay.tmp.png";
    if (!cv::imwrite(tmp_overlay.string(), overlay)) {
        throw std::runtime_error("无法写入 " + tmp_overlay.string());
    }
    std::error_code ec;
    if (fs::exists(overlay_path)) {
        fs::remove(overlay_path, ec);
        if (ec) {
            overlay_path = out_dir / "die_overlay_new.png";
        }
    }
    fs::rename(tmp_overlay, overlay_path, ec);
    if (ec) {
        fs::copy_file(tmp_overlay, overlay_path, fs::copy_options::overwrite_existing);
        fs::remove(tmp_overlay, ec);
    }
    return overlay_path;
}

void write_csv_row(std::ostream& csv, const DieBox& box, const std::string& file) {
    csv << box.x_idx << ',' << box.y_idx << ',' << box.left << ',' << box.top << ','
        << box.right << ',' << box.bottom << ',' << box.width() << ','
        << box.height() << ',' << (box.skipped ? 1 : 0) << ',' << file << "\r\n";
}

void print_cut_summary(
    const Options& opt,
    const DieMap& xml,
    int img_w,
    int img_h,
    double resolution_um,
    int lut,
    double cx,
    double cy,
    double radius,
    const CutResult& result
) {
    std::cout << "图像: " << opt.image.filename().string() << ' ' << img_w << 'x' << img_h << '\n';
    std::cout << std::setprecision(15);
    std::cout << "分辨率: " << resolution_um << " um/px  LUT: " << lut << "px\n";
    std::cout.setf(std::ios::fixed);
    std::cout << "布局圆心: (" << std::setprecision(1) << cx << ", " << cy << ")  半径: " << radius << "px\n";
    std::cout << std::setprecision(6);
    std::cout << "die 尺寸: " << xml.pitch_x << "x" << xml.pitch_y << " mm  步进: "
              << (xml.pitch_x + xml.scribe_w) << "x" << (xml.pitch_y + xml.scribe_h) << " mm\n";
    std::cout << "原点偏移: (" << xml.origin_offset_x << ", " << xml.origin_offset_y << ") mm\n";
    std::cout << "有效 die: " << result.saved << "  跳过: " << result.skipped
              << "  输出: " << result.patch_dir.string() << '\n';
    std::cout << "预览: " << result.overlay_path.string() << '\n';
    std::cout << "清单: " << result.csv_path.string() << '\n';
}

}  // namespace

CutResult cut_patches(const Options& opt, double resolution_um) {
    if (opt.patch_out.empty()) {
        throw std::runtime_error("未指定 patch 输出目录");
    }
    if (resolution_um <= 0.0) {
        throw std::runtime_error("每个像素对应的 um 必须大于 0");
    }

    const DieMap xml = parse_die_map(opt.xml);
    std::cout << "  读取图像...\n";
    const cv::Mat full = cv::imread(opt.image.string(), cv::IMREAD_GRAYSCALE);
    if (full.empty()) {
        throw std::runtime_error("无法读取图像: " + opt.image.string());
    }
    const int img_h = full.rows;
    const int img_w = full.cols;

    double scale = 1.0;
    const cv::Mat detect = make_detect_image(full, scale, opt.overlay_preview_px);
    const int lut = detect_lut_width(full);
    const int wafer_w = img_w - lut;
    const auto boxes = generate_dies(xml, wafer_w, img_h, resolution_um, lut);
    const WaferCircle circle = make_wafer_circle(xml, img_w, img_h, resolution_um, lut);

    CutResult result;
    result.patch_dir = opt.patch_out;
    result.full = full;
    result.boxes = boxes;
    fs::create_directories(opt.out);
    fs::create_directories(result.patch_dir);
    clear_old_pngs(result.patch_dir);

    const double inv = 1.0 / scale;
    const auto overlay = draw_overlay(
        detect, scale_boxes(boxes, inv), circle.cx * inv, circle.cy * inv, circle.radius * inv);
    result.overlay_path = save_overlay(overlay, opt.out);

    result.csv_path = opt.out / "die_map.csv";
    std::ofstream csv(result.csv_path, std::ios::binary);
    if (!csv) {
        throw std::runtime_error("无法写入 " + result.csv_path.string());
    }
    csv << "x_idx,y_idx,left,top,right,bottom,width,height,skipped,file\r\n";

    for (const auto& box : boxes) {
        const std::string name = box.file_name(opt.image.stem().string());
        if (box.skipped && !opt.keep_skipped) {
            ++result.skipped;
            write_csv_row(csv, box, "");
            continue;
        }
        const int left = std::max(box.left, 0);
        const int top = std::max(box.top, 0);
        const int right = std::min(box.right, img_w);
        const int bottom = std::min(box.bottom, img_h);
        if (right - left < 8 || bottom - top < 8) {
            continue;
        }
        const cv::Mat die = full(cv::Rect(left, top, right - left, bottom - top));
        const fs::path file_path = result.patch_dir / name;
        if (!cv::imwrite(file_path.string(), die)) {
            throw std::runtime_error("无法写入 " + file_path.string());
        }

        ++result.saved;
        result.skipped += box.skipped ? 1 : 0;
        write_csv_row(csv, box, name);
    }

    print_cut_summary(opt, xml, img_w, img_h, resolution_um, lut, circle.cx, circle.cy, circle.radius, result);
    return result;
}

}  // namespace cutdie
