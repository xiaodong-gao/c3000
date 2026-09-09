#pragma once

#include <filesystem>
#include <string>

namespace cutdie {

namespace fs = std::filesystem;

std::string to_lower(std::string s);

// Python 3 round()：四舍六入五成双
double py_round(double x);

// Python int()：向 0 截断
int py_int(double x);

bool is_png(const fs::path& path);
bool is_xml(const fs::path& path);

fs::path resolve_path(const fs::path& p, const fs::path& exe_dir);

}  // namespace cutdie
