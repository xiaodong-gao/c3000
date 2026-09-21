#include "die_util.h"

#include <cctype>
#include <cmath>

namespace cutdie {

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

double py_round(double x) {
    if (!std::isfinite(x)) {
        return x;
    }
    const double ax = std::fabs(x);
    double ip = 0.0;
    const double frac = std::modf(ax, &ip);
    double mag = ip;
    if (frac > 0.5) {
        mag = ip + 1.0;
    } else if (frac == 0.5) {
        if (std::fmod(ip, 2.0) != 0.0) {
            mag = ip + 1.0;
        }
    }
    return std::copysign(mag, x);
}

int py_int(double x) {
    return static_cast<int>(x);
}

bool is_png(const fs::path& path) {
    return to_lower(path.extension().string()) == ".png";
}

bool is_xml(const fs::path& path) {
    return to_lower(path.extension().string()) == ".xml";
}

fs::path resolve_path(const fs::path& p, const fs::path& exe_dir) {
    if (p.empty()) {
        return p;
    }
    if (p.is_absolute() && fs::exists(p)) {
        return p;
    }
    if (fs::exists(p)) {
        return fs::absolute(p);
    }
    const fs::path beside_exe = exe_dir / p;
    if (fs::exists(beside_exe)) {
        return beside_exe;
    }
    return p;
}

}  // namespace cutdie
