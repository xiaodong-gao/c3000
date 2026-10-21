#pragma once

#include "die_types.h"

namespace cutdie {

// 查找 recipes/<模式>.json。先从当前工作目录向上找，再从可执行文件目录向上找。
fs::path find_recipe(InspectMode mode);

// 用 json.hpp 读取配方。JSON 里没有的字段保持 Options 的默认值。
Options load_recipe(const fs::path& path);

// 用 json.hpp 把 Options 写成配方文件。
void save_recipe(const fs::path& path, const Options& opt);

}  // namespace cutdie
