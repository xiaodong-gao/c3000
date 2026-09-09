#include "die_inspect.h"

#include "die_util.h"

#include "EccAlign.cuh"
#include "HistogramThreshold.cuh"
#include "IntensityNormalize.cuh"
#include "PhaseCorrelate.cuh"
#include "Pipeline.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cutdie {
namespace {

const char* kImageExts[] = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pgm", ".ppm"};

bool is_image_file(const fs::path& path) {
    std::string ext = to_lower(path.extension().string());
    for (const char* allowed : kImageExts) {
        if (ext == allowed) {
            return true;
        }
    }
    return false;
}

bool parse_die_xy(const std::string& name, int& x, int& y) {
    const auto p = name.find("die_x");
    if (p == std::string::npos) {
        return false;
    }
    const char* s = name.c_str() + p + 5;
    char* end = nullptr;
    const long xv = std::strtol(s, &end, 10);
    if (end == s || end[0] != '_' || end[1] != 'y') {
        return false;
    }
    s = end + 2;
    const long yv = std::strtol(s, &end, 10);
    if (end == s) {
        return false;
    }
    x = static_cast<int>(xv);
    y = static_cast<int>(yv);
    return true;
}

const char* align_method_name(AlignMethod method) {
    switch (method) {
        case AlignMethod::EccCuda:
            return "ecc-cuda";
        case AlignMethod::EccCpu:
            return "ecc-cpu";
        case AlignMethod::PhaseCpu:
            return "phase-cpu";
        default:
            return "phase-cuda";
    }
}

PipelineBackend pipeline_backend_of(AlignMethod method) {
    return (method == AlignMethod::PhaseCpu || method == AlignMethod::EccCpu)
               ? PipelineBackend::Cpu
               : PipelineBackend::Cuda;
}

PipelineAlign pipeline_align_of(AlignMethod method) {
    return (method == AlignMethod::EccCpu || method == AlignMethod::EccCuda) ? PipelineAlign::Ecc
                                                                            : PipelineAlign::Phase;
}

PipelineBackend resolve_histogram_backend(HistogramMethod method, PipelineBackend align_backend) {
    if (method == HistogramMethod::Cpu) {
        return PipelineBackend::Cpu;
    }
    if (method == HistogramMethod::Cuda) {
        return PipelineBackend::Cuda;
    }
    return align_backend;
}

PipelineBackend resolve_normalize_backend(NormalizeMethod method, PipelineBackend align_backend) {
    if (method == NormalizeMethod::Cpu) {
        return PipelineBackend::Cpu;
    }
    if (method == NormalizeMethod::Cuda) {
        return PipelineBackend::Cuda;
    }
    return align_backend;
}

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

struct TemplateStore {
    TemplateCache cache;

    cv::Mat find(int x_idx, int y_idx, std::string& error, fs::path& used) const {
        used = cache.source;
        if (cache.single) {
            if (cache.single_img.empty()) {
                error = "外部模板为空: " + cache.source.string();
                return {};
            }
            return cache.single_img;
        }
        const auto it = cache.by_xy.find({x_idx, y_idx});
        if (it == cache.by_xy.end()) {
            error = "未找到 die_x" + std::to_string(x_idx) + "_y" + std::to_string(y_idx) + " 对应模板";
            return {};
        }
        return it->second;
    }
};

cv::Mat read_template_image(const fs::path& path) {
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("无法读取模板: " + path.string());
    }
    return img;
}

TemplateCache build_template_cache(const fs::path& template_path) {
    TemplateCache cache;
    cache.source = template_path;
    std::error_code ec;
    if (!fs::exists(template_path, ec)) {
        throw std::runtime_error("外部模板路径不存在: " + template_path.string());
    }
    if (fs::is_regular_file(template_path, ec)) {
        if (!is_image_file(template_path)) {
            throw std::runtime_error("外部模板不是图像: " + template_path.string());
        }
        cache.single = true;
        cache.single_img = read_template_image(template_path);
        return cache;
    }
    if (!fs::is_directory(template_path, ec)) {
        throw std::runtime_error("外部模板既不是文件也不是文件夹: " + template_path.string());
    }

    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(template_path, ec)) {
        if (ec) {
            throw std::runtime_error("无法读取模板文件夹: " + ec.message());
        }
        if (!entry.is_regular_file(ec) || !is_image_file(entry.path())) {
            continue;
        }
        images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) {
        throw std::runtime_error("模板文件夹中没有图像: " + template_path.string());
    }

    for (const auto& img : images) {
        int x = 0;
        int y = 0;
        if (parse_die_xy(img.stem().string(), x, y)) {
            cache.by_xy.emplace(std::make_pair(x, y), read_template_image(img));
        }
    }
    if (cache.by_xy.empty()) {
        if (images.size() == 1) {
            cache.single = true;
            cache.source = images.front();
            cache.single_img = read_template_image(images.front());
            return cache;
        }
        throw std::runtime_error(
            "模板文件夹有多张图，但文件名中没有 die_x*_y*，无法按 die 匹配: " +
            template_path.string());
    }
    return cache;
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

}  // namespace

TemplateCache load_templates(const fs::path& template_path) {
    return build_template_cache(template_path);
}

InspectResult inspect_wafer(const Options& opt, const CutResult& cut, const TemplateCache& templates) {
    if (cut.full.empty()) {
        throw std::runtime_error("切割结果中没有原始晶圆图");
    }
    if (templates.single ? templates.single_img.empty() : templates.by_xy.empty()) {
        throw std::runtime_error("外部模板为空");
    }

    TemplateStore store;
    store.cache = templates;
    const PipelineBackend align_backend = pipeline_backend_of(opt.align_method);
    const PipelineBackend hist_backend =
        resolve_histogram_backend(opt.histogram_method, align_backend);
    const PipelineBackend norm_backend =
        resolve_normalize_backend(opt.normalize_method, align_backend);
    const bool need_cuda = align_backend == PipelineBackend::Cuda ||
                           hist_backend == PipelineBackend::Cuda ||
                           norm_backend == PipelineBackend::Cuda;

    std::unique_ptr<PhaseCorrelateCuda> phase_cuda;
    std::unique_ptr<EccAlignCuda> ecc_cuda;
    std::unique_ptr<IntensityNormalizeCuda> intensity_cuda;
    if (need_cuda) {
        if (cudaSetDevice(0) != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice 失败");
        }
    }
    if (align_backend == PipelineBackend::Cuda) {
        phase_cuda = std::make_unique<PhaseCorrelateCuda>();
        ecc_cuda = std::make_unique<EccAlignCuda>();
    }
    if (norm_backend == PipelineBackend::Cuda) {
        intensity_cuda = std::make_unique<IntensityNormalizeCuda>();
    }

    HistogramThreshold histogram_threshold(opt.histogram);
    PipelineContext ctx;
    ctx.hist = &histogram_threshold;
    ctx.phaseCuda = phase_cuda.get();
    ctx.eccCuda = ecc_cuda.get();
    ctx.intensityCuda = intensity_cuda.get();
    ctx.defectParams = opt.defect;
    ctx.eccParams = opt.ecc;
    ctx.overrideHistogramBackend = true;
    ctx.histogramBackend = hist_backend;
    ctx.overrideNormalizeBackend = true;
    ctx.normalizeBackend = norm_backend;
    if (ecc_cuda) {
        EccCudaParams cuda_ecc{};
        cuda_ecc.maxIters = opt.ecc.maxIters;
        cuda_ecc.eps = opt.ecc.eps;
        cuda_ecc.gaussFiltSize = opt.ecc.gaussFiltSize;
        ecc_cuda->setParams(cuda_ecc);
    }

    cv::Mat overlay = to_bgr(cut.full);
    const int preview_px = opt.overlay_preview_px > 0 ? opt.overlay_preview_px : kDetectTarget;
    const double scale = std::max(overlay.cols, overlay.rows) / static_cast<double>(preview_px);
    const int thickness = std::max(2, cvRound(std::max(1.0, scale)));
    const double font = std::max(0.35, 0.35 * std::max(1.0, scale));

    InspectResult result;
    result.csv_path = opt.out / "defects.csv";
    result.overlay_path = opt.out / "defect_overlay.png";
    fs::create_directories(opt.out);

    std::cout << "  检出: 模板=" << templates.source.string()
              << (templates.single ? " (单模板)" : " (按 die 匹配)") << '\n';
    std::cout << "  检出后端: align=" << align_method_name(opt.align_method)
              << "  histogram=" << pipelineBackendName(hist_backend)
              << "  normalize=" << pipelineBackendName(norm_backend) << '\n';

    std::vector<std::vector<cv::Point>> all_contours;
    for (const auto& box : cut.boxes) {
        if (box.skipped && !opt.keep_skipped) {
            ++result.skipped_die;
            continue;
        }

        cv::Rect crop;
        cv::Mat data = crop_die(cut.full, box, crop);
        if (data.empty()) {
            ++result.skipped_die;
            continue;
        }

        std::string tmpl_error;
        fs::path tmpl_used;
        cv::Mat templ = store.find(box.x_idx, box.y_idx, tmpl_error, tmpl_used);
        if (templ.empty()) {
            std::cerr << "    跳过 die_x" << box.x_idx << "_y" << box.y_idx << ": " << tmpl_error
                      << '\n';
            ++result.skipped_no_template;
            cv::rectangle(overlay, crop, cv::Scalar(80, 80, 255), thickness);
            continue;
        }

        if (!match_pair_size(templ, data)) {
            std::cerr << "    跳过 die_x" << box.x_idx << "_y" << box.y_idx
                      << ": 与模板重叠区域过小\n";
            ++result.failed;
            continue;
        }

        const PipelineOutput pipe = runD2DPipeline(
            templ, data, align_backend, pipeline_align_of(opt.align_method), ctx);
        if (!pipe.ok) {
            std::cerr << "    失败 die_x" << box.x_idx << "_y" << box.y_idx << ": " << pipe.error
                      << '\n';
            ++result.failed;
            cv::rectangle(overlay, crop, cv::Scalar(0, 165, 255), thickness);
            continue;
        }

        ++result.inspected;
        const cv::Scalar die_color =
            pipe.candidates.empty() ? cv::Scalar(90, 255, 0) : cv::Scalar(0, 200, 255);
        cv::rectangle(overlay, crop, die_color, thickness);

        const int origin_x = crop.x + pipe.commonRoi.x;
        const int origin_y = crop.y + pipe.commonRoi.y;
        for (const DefectCandidate& c : pipe.candidates) {
            WaferDefect d;
            d.die_x = box.x_idx;
            d.die_y = box.y_idx;
            d.id = c.id;
            d.bbox = cv::Rect(origin_x + c.bbox.x, origin_y + c.bbox.y, c.bbox.width, c.bbox.height);
            d.bbox &= cv::Rect(0, 0, overlay.cols, overlay.rows);
            d.centroid = cv::Point2d(origin_x + c.centroid.x, origin_y + c.centroid.y);
            d.area = c.area;
            d.meanL2 = c.meanL2;
            d.maxL2 = c.maxL2;
            if (d.bbox.width <= 0 || d.bbox.height <= 0) {
                continue;
            }
            result.defects.push_back(d);
            cv::rectangle(overlay, d.bbox, cv::Scalar(0, 0, 255), std::max(2, thickness));
            cv::putText(overlay,
                        std::to_string(box.x_idx) + "," + std::to_string(box.y_idx) + "#" +
                            std::to_string(c.id),
                        cv::Point(d.bbox.x, std::max(12, d.bbox.y - 4)), cv::FONT_HERSHEY_SIMPLEX,
                        font, cv::Scalar(0, 0, 255), std::max(1, thickness / 2), cv::LINE_AA);
        }

        if (!pipe.blobMask.empty()) {
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(pipe.blobMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (auto& contour : contours) {
                for (auto& pt : contour) {
                    pt.x += origin_x;
                    pt.y += origin_y;
                }
                all_contours.push_back(std::move(contour));
            }
        }
    }

    if (!all_contours.empty()) {
        cv::drawContours(overlay, all_contours, -1, cv::Scalar(0, 0, 255),
                         std::max(1, thickness / 2));
    }
    result.defect_count = static_cast<int>(result.defects.size());
    result.overlay_path = save_png(overlay, result.overlay_path);

    std::ofstream csv(result.csv_path, std::ios::binary);
    if (!csv) {
        throw std::runtime_error("无法写入 " + result.csv_path.string());
    }
    csv << "die_x,die_y,id,x,y,w,h,area,cx,cy,meanL2,maxL2\r\n";
    for (const auto& d : result.defects) {
        csv << d.die_x << ',' << d.die_y << ',' << d.id << ',' << d.bbox.x << ',' << d.bbox.y << ','
            << d.bbox.width << ',' << d.bbox.height << ',' << d.area << ',' << d.centroid.x << ','
            << d.centroid.y << ',' << d.meanL2 << ',' << d.maxL2 << "\r\n";
    }

    std::cout << "  检出完成: 成功 " << result.inspected << "  失败 " << result.failed
              << "  无模板 " << result.skipped_no_template << "  跳过 " << result.skipped_die
              << "  缺陷 " << result.defect_count << '\n';
    std::cout << "  缺陷原图: " << result.overlay_path.string() << '\n';
    std::cout << "  缺陷清单: " << result.csv_path.string() << '\n';
    return result;
}

}  // namespace cutdie
