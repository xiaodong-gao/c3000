#include "die_inspect.h"

#include "die_util.h"

#include "EccAlign.cuh"
#include "HistogramThreshold.cuh"
#include "IntensityNormalize.cuh"
#include "PhaseCorrelate.hpp"
#include "PhaseCorrelate.cuh"
#include "Pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
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

std::string trim_copy(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
        ++i;
    }
    return s.substr(i);
}

bool d2d_enabled(const Options& opt) {
    return opt.defect.filter.keepBright || opt.defect.filter.keepDark ||
           opt.defect.filter.keepMixed || opt.defect.filter.keepUnknown;
}

void merge_large_white(const cv::Mat& data, const Options& opt, PipelineOutput& pipe) {
    if (data.empty()) {
        return;
    }
    std::vector<DefectCandidate> crop_cands;
    crop_cands.reserve(pipe.candidates.size() + 8);
    for (DefectCandidate c : pipe.candidates) {
        c.bbox.x += pipe.commonRoi.x;
        c.bbox.y += pipe.commonRoi.y;
        c.centroid.x += pipe.commonRoi.x;
        c.centroid.y += pipe.commonRoi.y;
        crop_cands.push_back(c);
    }
    const auto append_unique = [&](const std::vector<DefectCandidate>& extra, DefectPolarity pol) {
        for (const auto& w : extra) {
            bool overlap = false;
            for (const auto& e : crop_cands) {
                if (e.polarity != pol) {
                    continue;
                }
                const cv::Rect hit = e.bbox & w.bbox;
                const int need = std::max(1, static_cast<int>(0.3 * w.bbox.area()));
                if (hit.area() >= need) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                crop_cands.push_back(w);
            }
        }
    };
    if (opt.defect.largeWhite.enable) {
        append_unique(extractLargeWhiteRegions(data, opt.defect.largeWhite), DefectPolarity::Bright);
    }
    if (opt.defect.darkRegion.enable) {
        append_unique(extractDarkRegions(data, opt.defect.darkRegion), DefectPolarity::Dark);
    }
    int rec_id = 0;
    for (auto& c : crop_cands) {
        c.id = ++rec_id;
    }
    pipe.candidates.swap(crop_cands);
    pipe.defectOverlay = drawDefectOverlay(data, pipe.candidates);
}

std::set<std::string> load_truth_names(const fs::path& path) {
    std::set<std::string> truth;
    std::error_code ec;
    if (fs::is_regular_file(path, ec) && to_lower(path.extension().string()) == ".txt") {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("无法读取正样本列表: " + path.string());
        }
        std::string line;
        while (std::getline(in, line)) {
            line = trim_copy(line);
            if (line.empty()) {
                continue;
            }
            truth.insert(fs::path(line).filename().string());
        }
        return truth;
    }
    if (!fs::is_directory(path, ec)) {
        throw std::runtime_error("正样本目录不存在: " + path.string());
    }
    for (const auto& entry : fs::directory_iterator(path, ec)) {
        if (!entry.is_regular_file(ec) || !is_image_file(entry.path())) {
            continue;
        }
        truth.insert(entry.path().filename().string());
    }
    return truth;
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

cv::Mat to_gray_u8(const cv::Mat& src) {
    if (src.empty()) {
        return {};
    }
    if (src.channels() == 1) {
        if (src.type() == CV_8UC1) {
            return src;
        }
        cv::Mat u8;
        src.convertTo(u8, CV_8U);
        return u8;
    }
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return {};
    }
    return gray;
}

double appearance_mae(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat ga = to_gray_u8(a);
    cv::Mat gb = to_gray_u8(b);
    if (ga.empty() || gb.empty()) {
        return 1e9;
    }
    const int w = std::min(ga.cols, gb.cols);
    const int h = std::min(ga.rows, gb.rows);
    if (w < 8 || h < 8) {
        return 1e9;
    }
    cv::Mat da = ga(cv::Rect(0, 0, w, h));
    cv::Mat db = gb(cv::Rect(0, 0, w, h));
    cv::Mat diff;
    cv::absdiff(da, db, diff);
    return cv::mean(diff)[0];
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

    std::vector<cv::Mat> find_refs(int x_idx, int y_idx, const cv::Mat& data, bool neighbor, int want,
                                   std::string& error) const {
        std::vector<cv::Mat> refs;
        if (cache.single || !neighbor) {
            fs::path used;
            cv::Mat one = find(x_idx, y_idx, error, used);
            if (!one.empty()) {
                refs.push_back(one);
            }
            return refs;
        }
        if (cache.by_xy.empty()) {
            error = "文件夹模板为空";
            return refs;
        }

        struct Scored {
            double mae = 0.0;
            cv::Mat img;
        };
        std::vector<Scored> scored;
        static const int kOff[][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-2, 0}, {2, 0},
                                      {0, -2}, {0, 2}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1},
                                      {-3, 0}, {3, 0}, {0, -3}, {0, 3}};
        for (const auto& off : kOff) {
            const auto it = cache.by_xy.find({x_idx + off[0], y_idx + off[1]});
            if (it == cache.by_xy.end() || it->second.empty()) {
                continue;
            }
            scored.push_back({appearance_mae(it->second, data), it->second});
        }
        if (scored.empty()) {
            error = "未找到 die_x" + std::to_string(x_idx) + "_y" + std::to_string(y_idx) + " 的相邻模板";
            return refs;
        }
        std::sort(scored.begin(), scored.end(),
                  [](const Scored& a, const Scored& b) { return a.mae < b.mae; });
        const int n = std::max(1, std::min(want, static_cast<int>(scored.size())));
        for (int i = 0; i < n; ++i) {
            refs.push_back(scored[static_cast<size_t>(i)].img);
        }
        return refs;
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

TemplateCache templates_from_cut(const CutResult& cut, bool keep_skipped) {
    TemplateCache cache;
    cache.single = false;
    cache.source = "wafer-neighbor";
    for (const auto& box : cut.boxes) {
        if (box.skipped && !keep_skipped) {
            continue;
        }
        cv::Rect crop;
        cv::Mat img = crop_die(cut.full, box, crop);
        if (img.empty()) {
            continue;
        }
        cache.by_xy.emplace(std::make_pair(box.x_idx, box.y_idx), std::move(img));
    }
    if (cache.by_xy.empty()) {
        throw std::runtime_error("切割结果中没有可用于 D2D 的 die");
    }
    return cache;
}

InspectResult inspect_wafer(const Options& opt, const CutResult& cut, const TemplateCache& templates) {
    if (cut.full.empty()) {
        throw std::runtime_error("切割结果中没有原始晶圆图");
    }
    const bool run_d2d = opt.defect.filter.keepBright || opt.defect.filter.keepDark ||
                         opt.defect.filter.keepMixed || opt.defect.filter.keepUnknown;
    if (run_d2d && (templates.single ? templates.single_img.empty() : templates.by_xy.empty())) {
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
    PhaseCorrelate phase;
    PipelineContext ctx;
    ctx.hist = &histogram_threshold;
    ctx.phase = &phase;
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
        cuda_ecc.motionType = opt.ecc.motionType;
        ecc_cuda->setParams(cuda_ecc);
    }

    cv::Mat overlay = to_bgr(cut.full);
    const int preview_px = opt.overlay_preview_px > 0 ? opt.overlay_preview_px : kDetectTarget;
    const double scale = std::max(overlay.cols, overlay.rows) / static_cast<double>(preview_px);
    const int thickness = std::max(1, cvRound(scale));

    InspectResult result;
    result.csv_path = opt.out / "defects.csv";
    result.overlay_path = opt.out / "defect_overlay.png";
    fs::create_directories(opt.out);

    std::cout << "  检出: 模板=" << templates.source.string()
              << (templates.single ? " (单模板)"
                                   : (opt.template_neighbor ? " (相邻 die)" : " (按 die 匹配)"))
              << "  n=" << (templates.single ? 1 : static_cast<int>(templates.by_xy.size())) << '\n';
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

        PipelineOutput pipe;
        if (!run_d2d) {
            pipe.ok = true;
        } else {
            std::string tmpl_error;
            const int want = opt.template_neighbor ? std::max(1, opt.template_neighbor_min) : 1;
            std::vector<cv::Mat> refs =
                store.find_refs(box.x_idx, box.y_idx, data, opt.template_neighbor, want, tmpl_error);
            if (refs.empty()) {
                std::cerr << "    跳过 die_x" << box.x_idx << "_y" << box.y_idx << ": " << tmpl_error
                          << '\n';
                ++result.skipped_no_template;
                cv::rectangle(overlay, crop, cv::Scalar(80, 80, 255), thickness);
                continue;
            }

            int neighbor_hits = 0;
            std::string pipe_error;
            for (cv::Mat templ : refs) {
                cv::Mat data_one = data.clone();
                if (!match_pair_size(templ, data_one)) {
                    continue;
                }
                PipelineOutput one = runD2DPipeline(
                    templ, data_one, align_backend, pipeline_align_of(opt.align_method), ctx);
                if (!one.ok) {
                    pipe_error = one.error;
                    continue;
                }
                if (!one.candidates.empty()) {
                    ++neighbor_hits;
                }
                if (!pipe.ok || one.candidates.size() > pipe.candidates.size()) {
                    pipe = std::move(one);
                }
            }
            if (!pipe.ok) {
                std::cerr << "    失败 die_x" << box.x_idx << "_y" << box.y_idx << ": "
                          << (pipe_error.empty() ? "与模板重叠区域过小" : pipe_error) << '\n';
                ++result.failed;
                cv::rectangle(overlay, crop, cv::Scalar(0, 165, 255), thickness);
                continue;
            }
            if (opt.template_neighbor && opt.template_neighbor_min >= 2 &&
                static_cast<int>(refs.size()) >= 2 && neighbor_hits < opt.template_neighbor_min) {
                std::vector<DefectCandidate> keep;
                const int minBright = std::max(400, opt.defect.blob.minBrightArea);
                for (const auto& c : pipe.candidates) {
                    if (c.polarity == DefectPolarity::Bright && c.area >= minBright) {
                        keep.push_back(c);
                    }
                }
                pipe.candidates.swap(keep);
            }
        }

        std::vector<DefectCandidate> crop_cands;
        crop_cands.reserve(pipe.candidates.size() + 4);
        for (DefectCandidate c : pipe.candidates) {
            c.bbox.x += pipe.commonRoi.x;
            c.bbox.y += pipe.commonRoi.y;
            c.centroid.x += pipe.commonRoi.x;
            c.centroid.y += pipe.commonRoi.y;
            crop_cands.push_back(c);
        }
        if (opt.defect.largeWhite.enable) {
            const std::vector<DefectCandidate> whites =
                extractLargeWhiteRegions(data, opt.defect.largeWhite);
            for (const auto& w : whites) {
                bool overlap = false;
                for (const auto& e : crop_cands) {
                    if (e.polarity != DefectPolarity::Bright) {
                        continue;
                    }
                    const cv::Rect hit = e.bbox & w.bbox;
                    const int need = std::max(1, static_cast<int>(0.3 * w.bbox.area()));
                    if (hit.area() >= need) {
                        overlap = true;
                        break;
                    }
                }
                if (!overlap) {
                    crop_cands.push_back(w);
                }
            }
        }
        if (opt.defect.darkRegion.enable) {
            const std::vector<DefectCandidate> darks =
                extractDarkRegions(data, opt.defect.darkRegion);
            for (const auto& w : darks) {
                bool overlap = false;
                for (const auto& e : crop_cands) {
                    if (e.polarity != DefectPolarity::Dark) {
                        continue;
                    }
                    const cv::Rect hit = e.bbox & w.bbox;
                    const int need = std::max(1, static_cast<int>(0.3 * w.bbox.area()));
                    if (hit.area() >= need) {
                        overlap = true;
                        break;
                    }
                }
                if (!overlap) {
                    crop_cands.push_back(w);
                }
            }
        }
        int rec_id = 0;
        for (auto& c : crop_cands) {
            c.id = ++rec_id;
        }

        ++result.inspected;
        if (result.inspected % 50 == 0) {
            std::cout << "    检出进度 " << result.inspected << "/" << cut.boxes.size()
                      << "  缺陷 " << result.defects.size() << '\n';
        }
        const cv::Scalar die_color =
            crop_cands.empty() ? cv::Scalar(90, 255, 0) : cv::Scalar(0, 200, 255);
        cv::rectangle(overlay, crop, die_color, thickness);

        const int origin_x = crop.x;
        const int origin_y = crop.y;
        int min_vis = std::max(crop.width, crop.height) / 4;
        if (min_vis < 6) {
            min_vis = 6;
        }
        if (min_vis > 24) {
            min_vis = 24;
        }
        const int mark_r = 1;
        const cv::Rect wafer_bounds(0, 0, overlay.cols, overlay.rows);

        if (!crop_cands.empty()) {
            if (result.die_overlay_dir.empty()) {
                result.die_overlay_dir = opt.out / "die_defects";
                fs::create_directories(result.die_overlay_dir);
            }
            cv::Mat die_view = drawDefectOverlay(data, crop_cands);
            save_png(die_view, result.die_overlay_dir / box.file_name(opt.image.stem().string()));
        }

        for (const DefectCandidate& c : crop_cands) {
            WaferDefect d;
            d.die_x = box.x_idx;
            d.die_y = box.y_idx;
            d.id = c.id;
            d.bbox = cv::Rect(origin_x + c.bbox.x, origin_y + c.bbox.y, c.bbox.width, c.bbox.height);
            d.bbox &= wafer_bounds;
            d.centroid = cv::Point2d(origin_x + c.centroid.x, origin_y + c.centroid.y);
            d.area = c.area;
            d.meanL2 = c.meanL2;
            d.maxL2 = c.maxL2;
            d.meanSigned = c.meanSigned;
            d.brightArea = c.brightArea;
            d.darkArea = c.darkArea;
            d.polarity = c.polarity;
            if (d.bbox.width <= 0 || d.bbox.height <= 0) {
                continue;
            }
            result.defects.push_back(d);
            cv::Scalar color(0, 0, 255);
            if (c.polarity == DefectPolarity::Dark) {
                color = cv::Scalar(255, 80, 0);
            } else if (c.polarity == DefectPolarity::Mixed) {
                color = cv::Scalar(0, 200, 255);
            } else if (c.polarity == DefectPolarity::Unknown) {
                color = cv::Scalar(180, 180, 180);
            }
            cv::Rect vis = d.bbox;
            if (vis.width < min_vis || vis.height < min_vis) {
                const int cx = vis.x + vis.width / 2;
                const int cy = vis.y + vis.height / 2;
                vis.x = cx - min_vis / 2;
                vis.y = cy - min_vis / 2;
                vis.width = min_vis;
                vis.height = min_vis;
                vis &= wafer_bounds;
            }
            cv::rectangle(overlay, vis.area() > 0 ? vis : d.bbox, color, 1);
            const cv::Point center(cvRound(d.centroid.x), cvRound(d.centroid.y));
            if (wafer_bounds.contains(center)) {
                cv::circle(overlay, center, mark_r, color, cv::FILLED, cv::LINE_AA);
            }
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
    csv << "die_x,die_y,id,polarity,x,y,w,h,area,cx,cy,meanL2,maxL2,meanSigned,brightArea,darkArea\r\n";
    for (const auto& d : result.defects) {
        csv << d.die_x << ',' << d.die_y << ',' << d.id << ',' << defectPolarityName(d.polarity)
            << ',' << d.bbox.x << ',' << d.bbox.y << ',' << d.bbox.width << ',' << d.bbox.height
            << ',' << d.area << ',' << d.centroid.x << ',' << d.centroid.y << ',' << d.meanL2 << ','
            << d.maxL2 << ',' << d.meanSigned << ',' << d.brightArea << ',' << d.darkArea
            << "\r\n";
    }

    std::cout << "  检出完成: 成功 " << result.inspected << "  失败 " << result.failed
              << "  无模板 " << result.skipped_no_template << "  跳过 " << result.skipped_die
              << "  缺陷 " << result.defect_count << '\n';
    std::cout << "  缺陷原图: " << result.overlay_path.string() << '\n';
    std::cout << "  单颗 die 标注: " << result.die_overlay_dir.string() << '\n';
    std::cout << "  缺陷清单: " << result.csv_path.string() << '\n';
    return result;
}

PatchEvalResult eval_patches(const Options& opt, const TemplateCache& templates) {
    if (opt.eval_patch_dir.empty()) {
        throw std::runtime_error("eval_patch_dir 为空");
    }
    if (opt.eval_truth_dir.empty()) {
        throw std::runtime_error("eval_truth_dir 为空");
    }
    const bool run_d2d = d2d_enabled(opt);
    if (run_d2d && (templates.single ? templates.single_img.empty() : templates.by_xy.empty())) {
        throw std::runtime_error("外部模板为空");
    }

    std::error_code ec;
    if (!fs::is_directory(opt.eval_patch_dir, ec)) {
        throw std::runtime_error("待测 patch 目录不存在: " + opt.eval_patch_dir.string());
    }

    const std::set<std::string> truth = load_truth_names(opt.eval_truth_dir);
    if (truth.empty()) {
        throw std::runtime_error("正样本列表为空: " + opt.eval_truth_dir.string());
    }

    std::vector<fs::path> patches;
    const bool truth_is_list =
        fs::is_regular_file(opt.eval_truth_dir, ec) &&
        to_lower(opt.eval_truth_dir.extension().string()) == ".txt";
    if (truth_is_list) {
        for (const auto& name : truth) {
            patches.push_back(opt.eval_patch_dir / name);
        }
    } else {
        for (const auto& entry : fs::directory_iterator(opt.eval_patch_dir, ec)) {
            if (!entry.is_regular_file(ec) || !is_image_file(entry.path())) {
                continue;
            }
            patches.push_back(entry.path());
        }
    }
    std::sort(patches.begin(), patches.end());
    if (patches.empty()) {
        throw std::runtime_error("待测 patch 目录中没有图像: " + opt.eval_patch_dir.string());
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
    PhaseCorrelate phase;
    PipelineContext ctx;
    ctx.hist = &histogram_threshold;
    ctx.phase = &phase;
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
        cuda_ecc.motionType = opt.ecc.motionType;
        ecc_cuda->setParams(cuda_ecc);
    }

    PatchEvalResult result;
    result.total = static_cast<int>(patches.size());
    result.overlay_dir = opt.out.empty() ? (opt.eval_patch_dir.parent_path() / "patch_eval")
                                         : opt.out;
    result.report_path = result.overlay_dir / "eval_report.csv";
    fs::create_directories(result.overlay_dir / "fn");
    fs::create_directories(result.overlay_dir / "fp");
    fs::create_directories(result.overlay_dir / "tp");

    std::ofstream csv(result.report_path, std::ios::binary);
    if (!csv) {
        throw std::runtime_error("无法写入 " + result.report_path.string());
    }
    csv << "file,gt,hit,tag,n,area,meanL2,maxL2,meanSigned,bestMaxL2,bestMeanL2,bestAbsSigned,"
           "polarity,error\r\n";
    const fs::path cand_path = result.overlay_dir / "eval_candidates.csv";
    std::ofstream cand(cand_path, std::ios::binary);
    if (!cand) {
        throw std::runtime_error("无法写入 " + cand_path.string());
    }
    cand << "file,gt,id,polarity,area,w,h,meanL2,maxL2,meanSigned,brightArea,darkArea\r\n";

    std::cout << "  patch 评测: n=" << patches.size() << "  正样本=" << truth.size()
              << "  模板=" << templates.source.string()
              << (opt.template_neighbor ? " (相邻 die)" : "") << '\n';

    std::vector<std::string> fn_names;
    std::vector<std::string> fp_names;
    int done = 0;
    for (const auto& path : patches) {
        ++done;
        const std::string name = path.filename().string();
        const bool gt = truth.count(name) != 0;

        int x_idx = 0;
        int y_idx = 0;
        parse_die_xy(path.stem().string(), x_idx, y_idx);

        std::string tmpl_error;
        cv::Mat data = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        std::string error;
        PipelineOutput pipe;
        if (data.empty()) {
            error = "无法读取 " + path.string();
        } else if (!run_d2d) {
            pipe.ok = true;
            merge_large_white(data, opt, pipe);
        } else {
            const int want = opt.template_neighbor ? std::max(1, opt.template_neighbor_min) : 1;
            std::vector<cv::Mat> refs =
                store.find_refs(x_idx, y_idx, data, opt.template_neighbor, want, tmpl_error);
            if (refs.empty()) {
                error = tmpl_error;
            } else {
                int neighbor_hits = 0;
                std::string pipe_error;
                for (cv::Mat templ : refs) {
                    cv::Mat data_one = data.clone();
                    if (!match_pair_size(templ, data_one)) {
                        continue;
                    }
                    PipelineOutput one = runD2DPipeline(
                        templ, data_one, align_backend, pipeline_align_of(opt.align_method), ctx);
                    if (!one.ok) {
                        pipe_error = one.error;
                        continue;
                    }
                    if (!one.candidates.empty()) {
                        ++neighbor_hits;
                    }
                    if (!pipe.ok || one.candidates.size() > pipe.candidates.size()) {
                        pipe = std::move(one);
                    }
                }
                if (!pipe.ok) {
                    error = pipe_error.empty() ? "与模板重叠过小" : pipe_error;
                } else if (opt.template_neighbor && opt.template_neighbor_min >= 2 &&
                           static_cast<int>(refs.size()) >= 2 &&
                           neighbor_hits < opt.template_neighbor_min) {
                    pipe.candidates.clear();
                }
            }
            if (error.empty()) {
                merge_large_white(data, opt, pipe);
            }
        }

        const int n = static_cast<int>(pipe.candidates.size());
        const bool hit = error.empty() && n > 0;
        int area = 0;
        double meanL2 = 0.0;
        double maxL2 = 0.0;
        double meanSigned = 0.0;
        double bestMaxL2 = 0.0;
        double bestMeanL2 = 0.0;
        double bestAbsSigned = 0.0;
        const char* polarity = "";
        for (const auto& c : pipe.candidates) {
            cand << name << ',' << (gt ? 1 : 0) << ',' << c.id << ',' << defectPolarityName(c.polarity)
                 << ',' << c.area << ',' << c.bbox.width << ',' << c.bbox.height << ',' << c.meanL2
                 << ',' << c.maxL2 << ',' << c.meanSigned << ',' << c.brightArea << ',' << c.darkArea
                 << "\r\n";
            bestMaxL2 = std::max(bestMaxL2, c.maxL2);
            bestMeanL2 = std::max(bestMeanL2, c.meanL2);
            bestAbsSigned = std::max(bestAbsSigned, std::abs(c.meanSigned));
            if (c.area > area) {
                area = c.area;
                meanL2 = c.meanL2;
                maxL2 = c.maxL2;
                meanSigned = c.meanSigned;
                polarity = defectPolarityName(c.polarity);
            }
        }

        std::string tag;
        if (!error.empty()) {
            ++result.failed;
            tag = gt ? "FN_FAIL" : "FAIL";
            if (gt) {
                ++result.fn;
                fn_names.push_back(name);
            }
        } else if (gt && hit) {
            ++result.tp;
            tag = "TP";
            if (!pipe.defectOverlay.empty()) {
                save_png(pipe.defectOverlay, result.overlay_dir / "tp" / name);
            }
        } else if (gt && !hit) {
            ++result.fn;
            tag = "FN";
            fn_names.push_back(name);
            const cv::Mat view = pipe.defectOverlay.empty() ? data : pipe.defectOverlay;
            save_png(view, result.overlay_dir / "fn" / name);
        } else if (!gt && hit) {
            ++result.fp;
            tag = "FP";
            fp_names.push_back(name);
            if (!pipe.defectOverlay.empty()) {
                save_png(pipe.defectOverlay, result.overlay_dir / "fp" / name);
            }
        } else {
            ++result.tn;
            tag = "TN";
        }

        csv << name << ',' << (gt ? 1 : 0) << ',' << (hit ? 1 : 0) << ',' << tag << ',' << n << ','
            << area << ',' << meanL2 << ',' << maxL2 << ',' << meanSigned << ',' << bestMaxL2 << ','
            << bestMeanL2 << ',' << bestAbsSigned << ',' << polarity << ',' << error << "\r\n";

        if (done % 50 == 0 || done == result.total) {
            std::cout << "    " << done << "/" << result.total << "  TP=" << result.tp
                      << " FN=" << result.fn << " FP=" << result.fp << " TN=" << result.tn
                      << '\n';
        }
    }

    const double recall =
        (result.tp + result.fn) > 0
            ? static_cast<double>(result.tp) / static_cast<double>(result.tp + result.fn)
            : 0.0;
    const double precision =
        (result.tp + result.fp) > 0
            ? static_cast<double>(result.tp) / static_cast<double>(result.tp + result.fp)
            : 0.0;
    std::cout << "  评测完成: TP=" << result.tp << " FN=" << result.fn << " FP=" << result.fp
              << " TN=" << result.tn << " fail=" << result.failed << '\n';
    std::cout << "  recall=" << recall << "  precision=" << precision << '\n';
    std::cout << "  报告: " << result.report_path.string() << '\n';
    if (!fn_names.empty()) {
        std::cout << "  漏检 FN:\n";
        for (const auto& n : fn_names) {
            std::cout << "    " << n << '\n';
        }
    }
    if (!fp_names.empty()) {
        std::cout << "  误检 FP (" << fp_names.size() << "):\n";
        const size_t show = std::min(fp_names.size(), static_cast<size_t>(30));
        for (size_t i = 0; i < show; ++i) {
            std::cout << "    " << fp_names[i] << '\n';
        }
        if (fp_names.size() > show) {
            std::cout << "    ...\n";
        }
    }
    return result;
}

}  // namespace cutdie
