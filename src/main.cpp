// 根据 DieMapSelect XML，从晶圆 PNG 中切割每一颗 die。
// 布局与 Python cut_die.py / C3000 DieMapLayout.CreateLayoutPoints 一致。

#include "die_app.h"

#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    try {
        cutdie::Options opt;
        opt.dir = "C:/Users/aspiring135/Desktop/datasets/50um-gray";
        opt.resolution_um = 50.0;  // 每个像素对应多少 um
        opt.out.clear();           // 空则写回输入文件夹
        opt.keep_skipped = false;
        opt.recursive = false;

        if (opt.resolution_um <= 0.0) {
            throw std::runtime_error("每个像素对应的 um 必须大于 0");
        }
        return cutdie::run(opt);
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << '\n';
        return 1;
    }
}
