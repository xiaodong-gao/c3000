#include "die_map.h"

#include "die_util.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cutdie {
namespace {

std::string read_text_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法读取文件: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::optional<std::string> find_attr(const std::string& text, const std::string& name) {
    const std::string key = name + "=\"";
    const auto p = text.find(key);
    if (p == std::string::npos) {
        return std::nullopt;
    }
    const auto begin = p + key.size();
    const auto e = text.find('"', begin);
    if (e == std::string::npos) {
        return std::nullopt;
    }
    return text.substr(begin, e - begin);
}

double attr_d(const std::string& text, const std::string& name, double fallback = 0.0) {
    const auto v = find_attr(text, name);
    return v ? std::stod(*v) : fallback;
}

int attr_i(const std::string& text, const std::string& name) {
    const auto v = find_attr(text, name);
    if (!v) {
        throw std::runtime_error("缺少属性 " + name);
    }
    return std::stoi(*v);
}

bool attr_b(const std::string& text, const std::string& name, bool fallback = false) {
    const auto v = find_attr(text, name);
    if (!v) {
        return fallback;
    }
    const std::string s = to_lower(*v);
    return s == "true" || s == "1" || s == "yes";
}

std::string extract_open_tag(const std::string& xml, const std::string& tag) {
    const std::string open = "<" + tag;
    const auto p = xml.find(open);
    if (p == std::string::npos) {
        throw std::runtime_error("XML 中缺少标签: " + tag);
    }
    const auto e = xml.find('>', p);
    if (e == std::string::npos) {
        throw std::runtime_error("XML 标签未闭合: " + tag);
    }
    return xml.substr(p, e - p + 1);
}

std::optional<double> parse_um(const std::string& text) {
    const std::string lower = to_lower(text);
    size_t pos = 0;
    while ((pos = lower.find("um", pos)) != std::string::npos) {
        size_t i = pos;
        while (i > 0 && (lower[i - 1] == ' ' || lower[i - 1] == '\t')) {
            --i;
        }
        const size_t num_end = i;
        size_t num_start = num_end;
        bool saw_dot = false;
        while (num_start > 0) {
            const char c = lower[num_start - 1];
            if (std::isdigit(static_cast<unsigned char>(c))) {
                --num_start;
            } else if (c == '.' && !saw_dot) {
                saw_dot = true;
                --num_start;
            } else {
                break;
            }
        }
        if (num_start < num_end) {
            return std::stod(text.substr(num_start, num_end - num_start));
        }
        pos += 2;
    }
    return std::nullopt;
}

struct LayoutGeom {
    double scale = 0;
    double center_x = 0;
    double center_y = 0;
    double die_w = 0;
    double die_h = 0;
    double step_x = 0;
    double step_y = 0;
    double offset_x = 0;
    double offset_y = 0;
    double max_dist2 = 0;
    int rect_w = 0;
    int rect_h = 0;
    double half_w = 0;
    double half_h = 0;
    int left_col = 0;
    int top_row = 0;
    int right_col = 0;
    int bottom_row = 0;
};

LayoutGeom make_layout_geom(const DieMap& xml, int img_w, int img_h, double resolution_um) {
    LayoutGeom g;
    g.scale = 1000.0 / resolution_um;
    g.center_x = img_w / 2.0 + xml.trim_cx * g.scale;
    g.center_y = img_h / 2.0 - xml.trim_cy * g.scale;
    g.die_w = xml.pitch_x;
    g.die_h = xml.pitch_y;
    g.step_x = g.die_w + xml.scribe_w;
    g.step_y = g.die_h + xml.scribe_h;
    g.offset_x = xml.origin_offset_x;
    g.offset_y = -xml.origin_offset_y;

    const double wafer_diameter = xml.wafer_diameter - xml.trim_size * 2.0;
    const double wafer_radius = (wafer_diameter - xml.edge_offset * 2.0) / 2.0;
    g.max_dist2 = wafer_radius * wafer_radius;
    g.rect_w = py_int(g.die_w * g.scale);
    g.rect_h = py_int(g.die_h * g.scale);
    g.half_w = g.die_w / 2.0;
    g.half_h = g.die_h / 2.0;

    g.left_col = g.die_w > 0.0 ? py_int(std::max(wafer_radius + g.offset_x, 0.0) / g.die_w) : 0;
    g.top_row = g.die_h > 0.0 ? py_int(std::max(wafer_radius + g.offset_y, 0.0) / g.die_h) : 0;
    g.right_col = g.die_w > 0.0 ? py_int(std::max(wafer_radius - g.offset_x, 0.0) / g.die_w) : 0;
    g.bottom_row = g.die_h > 0.0 ? py_int(std::max(wafer_radius - g.offset_y, 0.0) / g.die_h) : 0;
    return g;
}

bool die_corners_inside(double x_mm, double y_mm, const LayoutGeom& g) {
    const double corners[4][2] = {
        {-g.half_w, -g.half_h},
        {g.half_w, -g.half_h},
        {-g.half_w, g.half_h},
        {g.half_w, g.half_h},
    };
    for (const auto& c : corners) {
        const double dx = x_mm + c[0];
        const double dy = y_mm + c[1];
        if (dx * dx + dy * dy > g.max_dist2) {
            return false;
        }
    }
    return true;
}

std::optional<DieBox> make_die_box(
    const DieMap& xml,
    const LayoutGeom& g,
    int die_x,
    int die_y,
    int img_w,
    int img_h,
    int x_offset
) {
    const double x_mm = g.offset_x + die_x * g.step_x;
    const double y_mm = g.offset_y + die_y * g.step_y;
    if (!die_corners_inside(x_mm, y_mm, g)) {
        return std::nullopt;
    }

    const double x_px = py_round(x_mm * g.scale + g.center_x);
    const double y_px = py_round(y_mm * g.scale + g.center_y);
    const int left = py_int(x_px - g.rect_w / 2.0) + x_offset;
    const int top = py_int(y_px - g.rect_h / 2.0);
    const int right = left + g.rect_w;
    const int bottom = top + g.rect_h;
    if (right <= x_offset || bottom <= 0 || left >= img_w + x_offset || top >= img_h) {
        return std::nullopt;
    }

    DieBox box;
    box.x_idx = die_x;
    box.y_idx = -die_y;
    box.left = std::max(left, x_offset);
    box.top = std::max(top, 0);
    box.right = std::min(right, x_offset + img_w);
    box.bottom = std::min(bottom, img_h);
    if (box.width() < 8 || box.height() < 8) {
        return std::nullopt;
    }
    box.skipped =
        xml.skipped.count({die_x, die_y}) > 0 ||
        xml.skipped.count({box.x_idx, box.y_idx}) > 0;
    return box;
}

}  // namespace

DieMap parse_die_map(const fs::path& xml_path) {
    const std::string xml = read_text_file(xml_path);
    const std::string layout = extract_open_tag(xml, "PatternedWaferLayout");
    DieMap m;
    m.wafer_diameter = attr_d(layout, "WaferDiameter");
    m.edge_offset = attr_d(layout, "EdgeOffset");
    m.pitch_x = attr_d(layout, "DiePitchX");
    m.pitch_y = attr_d(layout, "DiePitchY");
    m.origin_offset_x = attr_d(layout, "OriginalDieOffsetX");
    m.origin_offset_y = attr_d(layout, "OriginalDieOffsetY");
    m.scribe_w = attr_d(layout, "ScribeLineWidth");
    m.scribe_h = attr_d(layout, "ScribeLineHeight");
    m.patterned = attr_b(layout, "Enabled", true);
    try {
        const std::string trimming = extract_open_tag(xml, "Trimming");
        m.trim_cx = attr_d(trimming, "CenterX");
        m.trim_cy = attr_d(trimming, "CenterY");
        m.trim_size = attr_d(trimming, "Size");
    } catch (const std::runtime_error&) {
        m.trim_cx = 0;
        m.trim_cy = 0;
        m.trim_size = 0;
    }
    try {
        const std::string gray = extract_open_tag(xml, "GrayValueRange");
        m.gray_min = static_cast<int>(attr_d(gray, "Min", 0.0));
        m.gray_max = static_cast<int>(attr_d(gray, "Max", 255.0));
        if (m.gray_min < 0) {
            m.gray_min = 0;
        }
        if (m.gray_max > 255) {
            m.gray_max = 255;
        }
        if (m.gray_max < m.gray_min) {
            std::swap(m.gray_min, m.gray_max);
        }
    } catch (const std::runtime_error&) {
        m.gray_min = 0;
        m.gray_max = 255;
    }

    const std::string skip_tag = "<SkippedDie";
    for (size_t p = 0; (p = xml.find(skip_tag, p)) != std::string::npos;) {
        const auto e = xml.find('>', p);
        if (e == std::string::npos) {
            break;
        }
        const std::string tag = xml.substr(p, e - p + 1);
        m.skipped.emplace(attr_i(tag, "XIdx"), attr_i(tag, "YIdx"));
        p = e + 1;
    }
    return m;
}

WaferCircle make_wafer_circle(const DieMap& xml, int img_w, int img_h, double resolution_um, int lut) {
    if (resolution_um <= 0.0) {
        throw std::runtime_error("每个像素对应的 um 必须大于 0");
    }
    if (img_w <= lut || img_h <= 0) {
        throw std::runtime_error("图像尺寸无法计算圆形 ROI");
    }
    const double px_per_mm = 1000.0 / resolution_um;
    const int wafer_w = img_w - lut;
    WaferCircle c;
    c.lut = lut;
    c.cx = lut + wafer_w / 2.0 + xml.trim_cx * px_per_mm;
    c.cy = img_h / 2.0 - xml.trim_cy * px_per_mm;
    // 圆形 ROI 用物理晶圆半径。EdgeOffset / TrimSize 只用于 die 是否落在片内，不能拿来缩小检测圆。
    const double wafer_radius_mm = xml.wafer_diameter / 2.0;
    if (wafer_radius_mm <= 0.0) {
        throw std::runtime_error("对齐配置中的晶圆半径无效");
    }
    c.radius = wafer_radius_mm * px_per_mm;
    return c;
}

double infer_resolution_um(const fs::path& path, double fallback) {
    const std::string texts[] = {
        path.filename().string(),
        path.parent_path().filename().string(),
        path.string(),
    };
    for (const auto& text : texts) {
        if (const auto v = parse_um(text)) {
            return *v;
        }
    }
    return fallback;
}

int detect_lut_width(const cv::Mat& gray) {
    const int max_w = std::min(200, gray.cols);
    const int rows = gray.rows;
    int width = 0;
    while (width < max_w) {
        double sum = 0.0;
        for (int y = 0; y < rows; ++y) {
            sum += gray.ptr<std::uint8_t>(y)[width];
        }
        if (sum / rows <= 100.0) {
            break;
        }
        ++width;
    }
    return width;
}

std::vector<DieBox> generate_dies(
    const DieMap& xml,
    int img_w,
    int img_h,
    double resolution_um,
    int x_offset
) {
    const LayoutGeom g = make_layout_geom(xml, img_w, img_h, resolution_um);
    std::vector<DieBox> boxes;
    for (int die_y = -g.top_row; die_y <= g.bottom_row; ++die_y) {
        for (int die_x = -g.left_col; die_x <= g.right_col; ++die_x) {
            if (auto box = make_die_box(xml, g, die_x, die_y, img_w, img_h, x_offset)) {
                boxes.push_back(*box);
            }
        }
    }
    std::sort(boxes.begin(), boxes.end(), [](const DieBox& a, const DieBox& b) {
        if (a.y_idx != b.y_idx) {
            return a.y_idx > b.y_idx;
        }
        return a.x_idx < b.x_idx;
    });
    return boxes;
}

std::vector<DieBox> scale_boxes(const std::vector<DieBox>& boxes, double scale) {
    if (scale == 1.0) {
        return boxes;
    }
    std::vector<DieBox> out;
    out.reserve(boxes.size());
    for (const auto& b : boxes) {
        DieBox n = b;
        n.left = py_int(py_round(b.left * scale));
        n.top = py_int(py_round(b.top * scale));
        n.right = py_int(py_round(b.right * scale));
        n.bottom = py_int(py_round(b.bottom * scale));
        out.push_back(n);
    }
    return out;
}

}  // namespace cutdie
