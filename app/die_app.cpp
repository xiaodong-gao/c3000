#include "die_app.h"

#include "die_cut.h"
#include "die_inspect.h"
#include "die_util.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace cutdie {
namespace {

bool is_cut_artifact_png(const fs::path& path) {
    const std::string name = to_lower(path.filename().string());
    if (name == "die_overlay.png" || name == "defect_overlay.png" ||
        name.find("defect_overlay") != std::string::npos || name.rfind("_dbg_", 0) == 0) {
        return true;
    }
    return name.find("die_x") != std::string::npos && name.find("_y") != std::string::npos;
}

bool is_skipped_dir(const fs::path& path) {
    const std::string name = to_lower(path.filename().string());
    static const char* kSkip[] = {
        "patches", "dies", "x64", "win32", "debug", "release", ".vs", ".git", "output",
        "test-patches"
    };
    for (const char* s : kSkip) {
        if (name == s) {
            return true;
        }
    }
    return false;
}

std::string xml_base_name(std::string stem) {
    const std::string lower = to_lower(stem);
    const std::string suffix = "diemapselect";
    if (lower.size() >= suffix.size() && lower.rfind(suffix) == lower.size() - suffix.size()) {
        stem.erase(stem.size() - suffix.size());
        while (!stem.empty() && (stem.back() == '_' || stem.back() == '-' || stem.back() == ' ')) {
            stem.pop_back();
        }
    }
    return stem;
}

void collect_files(
    const fs::path& dir,
    bool recursive,
    std::vector<fs::path>& pngs,
    std::vector<fs::path>& xmls) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("不是有效文件夹: " + dir.string());
    }
    const auto consider = [&](const fs::directory_entry& entry) {
        if (!entry.is_regular_file()) {
            return;
        }
        const fs::path& p = entry.path();
        if (is_png(p) && !is_cut_artifact_png(p)) {
            pngs.push_back(fs::absolute(p));
        } else if (is_xml(p)) {
            xmls.push_back(fs::absolute(p));
        }
    };
    if (recursive) {
        for (auto it = fs::recursive_directory_iterator(dir); it != fs::recursive_directory_iterator(); ++it) {
            if (it->is_directory() && is_skipped_dir(it->path())) {
                it.disable_recursion_pending();
                continue;
            }
            consider(*it);
        }
    } else {
        for (const auto& entry : fs::directory_iterator(dir)) {
            consider(entry);
        }
    }
    std::sort(pngs.begin(), pngs.end());
    std::sort(xmls.begin(), xmls.end());
}

int match_score(const std::string& png_stem, const std::string& xml_stem) {
    const std::string p = to_lower(png_stem);
    const std::string x = to_lower(xml_stem);
    const std::string xb = to_lower(xml_base_name(xml_stem));
    if (p == x || p == xb) {
        return 300;
    }
    if (x == p + "diemapselect" || x == p + "_diemapselect" || x == p + "-diemapselect") {
        return 280;
    }
    if (!xb.empty() && (p.find(xb) != std::string::npos || xb.find(p) != std::string::npos)) {
        return 100 + static_cast<int>(std::min(p.size(), xb.size()));
    }
    if (x.find(p) != std::string::npos || p.find(x) != std::string::npos) {
        return 80;
    }
    return 0;
}

fs::path find_xml_for_png(const fs::path& png, const std::vector<fs::path>& xmls) {
    const fs::path dir = png.parent_path();
    const std::string stem = png.stem().string();
    const std::vector<fs::path> exact = {
        dir / (stem + ".xml"),
        dir / (stem + "DieMapSelect.xml"),
        dir / (stem + "_DieMapSelect.xml"),
        dir / (stem + "-DieMapSelect.xml"),
    };
    for (const auto& c : exact) {
        if (fs::exists(c)) {
            return fs::absolute(c);
        }
    }
    int best = 0;
    fs::path best_xml;
    for (const auto& xml : xmls) {
        const int score = match_score(stem, xml.stem().string());
        if (score > best) {
            best = score;
            best_xml = xml;
        }
    }
    return best >= 100 ? best_xml : fs::path{};
}

Job make_job(const fs::path& image, const fs::path& xml, const fs::path& out) {
    Job job;
    job.image = image;
    job.xml = xml;
    job.name = image.stem().string();
    job.out = out;
    return job;
}

void resolve_job_outputs(Options& one, const Options& opt, const Job& job, size_t job_count) {
    one.image = job.image;
    one.xml = job.xml;
    one.out = job.out;
    if (!opt.patch_out.empty()) {
        one.patch_out = job_count == 1 ? opt.patch_out : (opt.patch_out / job.name);
    } else {
        one.patch_out = job.out / "patches";
    }
}

}  // namespace

std::vector<Job> collect_jobs(const Options& opt) {
    fs::path out_root = opt.out;
    if (!opt.image.empty()) {
        fs::path xml = opt.xml;
        if (xml.empty()) {
            std::vector<fs::path> xmls;
            std::vector<fs::path> pngs;
            collect_files(opt.image.parent_path(), false, pngs, xmls);
            xml = find_xml_for_png(opt.image, xmls);
        }
        if (xml.empty() || !fs::exists(xml)) {
            throw std::runtime_error("未找到与图像对应的 XML: " + opt.image.filename().string());
        }
        const fs::path out = out_root.empty()
            ? (opt.image.parent_path() / opt.image.stem())
            : (out_root / opt.image.stem());
        return {make_job(opt.image, xml, out)};
    }

    const fs::path dir = opt.dir.empty() ? fs::current_path() : opt.dir;
    std::vector<fs::path> pngs;
    std::vector<fs::path> xmls;
    collect_files(dir, opt.recursive, pngs, xmls);
    if (pngs.empty()) {
        throw std::runtime_error("文件夹中没有可切割的 PNG: " + dir.string());
    }
    if (xmls.empty()) {
        throw std::runtime_error("文件夹中没有 XML: " + dir.string());
    }
    if (out_root.empty()) {
        out_root = dir;
    }
    if (pngs.size() == 1 && xmls.size() == 1 && find_xml_for_png(pngs[0], xmls).empty()) {
        return {make_job(pngs[0], xmls[0], out_root / pngs[0].stem())};
    }

    std::vector<Job> jobs;
    for (const auto& png : pngs) {
        fs::path xml = find_xml_for_png(png, xmls);
        if (xml.empty()) {
            std::cerr << "跳过（无对应 XML）: " << png.filename().string() << '\n';
            continue;
        }
        jobs.push_back(make_job(png, xml, out_root / png.stem()));
    }
    if (jobs.empty()) {
        throw std::runtime_error("没有找到名字对应的 PNG/XML 配对");
    }
    return jobs;
}

int run(Options opt) {
    const fs::path base = fs::current_path();
    opt.dir = resolve_path(opt.dir, base);
    opt.image = resolve_path(opt.image, base);
    opt.xml = resolve_path(opt.xml, base);
    opt.patch_out = resolve_path(opt.patch_out, base);
    opt.template_path = resolve_path(opt.template_path, base);
    opt.eval_patch_dir = resolve_path(opt.eval_patch_dir, base);
    opt.eval_truth_dir = resolve_path(opt.eval_truth_dir, base);
    if (!opt.out.empty() && opt.out.is_relative()) {
        opt.out = fs::absolute(opt.out);
    }
    if (!opt.patch_out.empty() && opt.patch_out.is_relative()) {
        opt.patch_out = fs::absolute(opt.patch_out);
    }
    if (opt.resolution_um <= 0.0) {
        throw std::runtime_error("每个像素对应的 um 必须大于 0");
    }

    if (!opt.eval_patch_dir.empty()) {
        if (opt.template_path.empty()) {
            throw std::runtime_error("patch 评测需要指定外部模板");
        }
        if (opt.eval_truth_dir.empty()) {
            throw std::runtime_error("patch 评测需要指定 eval_truth_dir");
        }
        const TemplateCache templates = load_templates(opt.template_path);
        std::cout << "patch 评测模式\n";
        std::cout << "  待测: " << opt.eval_patch_dir.string() << '\n';
        std::cout << "  正样本: " << opt.eval_truth_dir.string() << '\n';
        std::cout << "  模板: " << opt.template_path.string() << '\n';
        const PatchEvalResult result = eval_patches(opt, templates);
        return (result.fn == 0 && result.fp == 0 && result.failed == 0) ? 0 : 1;
    }

    const double resolution_um = opt.resolution_um;
    const auto jobs = collect_jobs(opt);
    std::cout << "共 " << jobs.size() << " 组 PNG/XML 待切割\n";
    std::cout << "分辨率: " << resolution_um << " um/px\n";
    int failed = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto& job = jobs[i];
        Options one = opt;
        resolve_job_outputs(one, opt, job, jobs.size());
        std::cout << "\n[" << (i + 1) << "/" << jobs.size() << "] " << job.name << '\n';
        std::cout << "  PNG: " << job.image.string() << '\n';
        std::cout << "  XML: " << job.xml.string() << '\n';
        std::cout << "  输出: " << one.patch_out.string() << '\n';
        try {
            TemplateCache templates;
            if (one.inspect && !one.template_path.empty()) {
                templates = load_templates(one.template_path);
                std::cout << "  模板: " << one.template_path.string()
                          << (templates.single ? " (单模板已预载)" : " (按 die 匹配已预载)")
                          << '\n';
            }
            const CutResult cut = cut_patches(one, resolution_um);
            if (one.inspect) {
                if (one.template_neighbor && templates.by_xy.empty() && templates.single_img.empty()) {
                    templates = templates_from_cut(cut, one.keep_skipped);
                    std::cout << "  模板: 本片相邻 die (" << templates.by_xy.size() << ")\n";
                }
                if (templates.single ? templates.single_img.empty() : templates.by_xy.empty()) {
                    std::cerr << "  跳过检出: 未指定外部模板，且未启用相邻 die\n";
                } else {
                    inspect_wafer(one, cut, templates);
                }
            }
        } catch (const std::exception& ex) {
            std::cerr << "  失败: " << ex.what() << '\n';
            ++failed;
        }
    }
    if (failed > 0) {
        std::cerr << "\n完成，失败 " << failed << " / " << jobs.size() << '\n';
        return 1;
    }
    std::cout << "\n全部完成: " << jobs.size() << " 组\n";
    return 0;
}

}  // namespace cutdie

