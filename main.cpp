// 同一工程内切换 Single Die / Die-to-Die / Die-to-Database / yolo-openvino。
// SingleDie、DieToDie、DieToDatabase 的参数配方在 recipes/*.json，由 json.hpp 读入。
// 改 kInspectMode 后，main 只加载对应配方并调用 cutdie::run。
// 不使用命令行参数。

#include "die_app.h"
#include "recipe_json.h"

#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

// 有图形晶圆：切割 die patch，用 OpenVINO YOLO26 语义分割推理。
void apply_yolo_openvino_recipe(cutdie::Options& opt) {
    opt.mode = cutdie::InspectMode::YoloOpenVino;

    opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-0804";
    opt.image = "C:/Users/aspiring135/Desktop/datasets/50um-0804/0804-slot1.png";
    opt.xml = "C:/Users/aspiring135/Desktop/datasets/50um-0804/0804-slot1.xml";
    opt.out.clear();
    opt.patch_out.clear();
    opt.resolution_um = 50.0;
    opt.keep_skipped = false;
    opt.recursive = false;
    opt.overlay_preview_px = 8000;
    opt.inspect = true;

    opt.yolo_model = "C:/Users/aspiring135/Desktop/project/c3000/models/best.xml";
    opt.yolo_device = "CPU";
    opt.yolo_imgsz = 64;
    opt.yolo_batch = 8;
    opt.yolo_requests = 0;
    opt.yolo_confidence = 0.25f;
    opt.yolo_fp16 = true;
    opt.yolo_throughput = false;

    opt.eval_patch_dir.clear();
    opt.eval_truth_dir.clear();
}

}  // namespace

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try {
        // ========== 切换检出模式 ==========
        // SingleDie     : recipes/SingleDie.json
        // DieToDie      : recipes/DieToDie.json
        // DieToDatabase : recipes/DieToDatabase.json
        // YoloOpenVino  : 切 die patch，OpenVINO YOLO26 语义分割（best.xml，64x64）

        constexpr cutdie::InspectMode kInspectMode = cutdie::InspectMode::DieToDie;

        cutdie::Options opt;
        if (kInspectMode == cutdie::InspectMode::YoloOpenVino) {
            apply_yolo_openvino_recipe(opt);
        } else if (kInspectMode == cutdie::InspectMode::AnomalibOpenVino) {
            throw std::runtime_error("AnomalibOpenVino 尚未配置");
        } else {
            const auto recipe = cutdie::find_recipe(kInspectMode);
            std::cout << "配方: " << recipe.string() << '\n';
            opt = cutdie::load_recipe(recipe);
            if (opt.mode != kInspectMode) {
                throw std::runtime_error("配方 mode 与 kInspectMode 不一致: " + recipe.string());
            }
        }

        if (opt.resolution_um <= 0.0) {
            throw std::runtime_error("每个像素对应的 um 必须大于 0");
        }
        return cutdie::run(opt);
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << '\n';
        return 1;
    }
}
