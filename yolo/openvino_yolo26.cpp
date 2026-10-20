#include "openvino_yolo26.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_cpu/properties.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace yolo26 {
namespace {

const cv::Scalar kLetterboxPad(114, 114, 114);

const std::array<cv::Scalar, 19> kCityColors = {
    cv::Scalar(128, 64, 128), cv::Scalar(232, 35, 244), cv::Scalar(70, 70, 70), cv::Scalar(156, 102, 102),
    cv::Scalar(153, 153, 190), cv::Scalar(153, 153, 153), cv::Scalar(30, 170, 250), cv::Scalar(0, 220, 220),
    cv::Scalar(35, 142, 107), cv::Scalar(152, 251, 152), cv::Scalar(180, 130, 70), cv::Scalar(60, 20, 220),
    cv::Scalar(0, 0, 255), cv::Scalar(142, 0, 0), cv::Scalar(70, 0, 0), cv::Scalar(100, 60, 0),
    cv::Scalar(100, 80, 0), cv::Scalar(230, 0, 0), cv::Scalar(32, 11, 119)
};

const std::array<cv::Vec3b, 256> kCityLut = [] {
    std::array<cv::Vec3b, 256> lut{};
    for (int i = 0; i < 256; ++i) {
        const cv::Scalar& color = kCityColors[static_cast<size_t>(i) % kCityColors.size()];
        lut[static_cast<size_t>(i)] = cv::Vec3b(static_cast<uint8_t>(color[0]),
                                                static_cast<uint8_t>(color[1]),
                                                static_cast<uint8_t>(color[2]));
    }
    return lut;
}();

struct Options {
    fs::path model;
    fs::path input;
    fs::path output = "output";
    fs::path labels;
    std::string device = "CPU";
    int width = 640;
    int height = 640;
    int batch = 1;
    int requests = 0;
    int opencv_threads = 0;
    float confidence = 0.25f;
    ov::element::Type inference_precision = ov::element::f16;
    bool throughput_mode = false;
};

struct Transform {
    float scale = 1.0f;
    float inv_scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int resized_w = 0;
    int resized_h = 0;
};

struct Detection {
    cv::Rect2f box;
    cv::Rect2f network_box;
    float score = 0.0f;
    int class_id = -1;
    float angle = 0.0f;
    std::array<cv::Point2f, 17> keypoints{};
    std::array<float, 17> keypoint_scores{};
    std::array<float, 64> mask_coefficients{};
    size_t mask_coefficient_count = 0;
};

// One heap buffer + null-terminated views. Names stay in memory for the whole run.
struct LabelTable {
    std::string blob;
    std::vector<const char*> names;

    const char* operator[](int id) const {
        if (id >= 0 && static_cast<size_t>(id) < names.size()) return names[static_cast<size_t>(id)];
        thread_local char fallback[32];
        std::snprintf(fallback, sizeof(fallback), "class_%d", id);
        return fallback;
    }

    size_t size() const { return names.size(); }
    bool empty() const { return names.empty(); }
};

std::string task_name(Task task) {
    switch (task) {
    case Task::Classification: return "classification";
    case Task::Depth: return "depth";
    case Task::Detection: return "detection";
    case Task::OrientedBoundingBox: return "obb";
    case Task::Pose: return "pose";
    case Task::InstanceSegmentation: return "instance-segmentation";
    case Task::SemanticSegmentation: return "semantic-segmentation";
    }
    return "unknown";
}

Options make_options(const Defaults& defaults) {
    Options options;
    options.model = defaults.model;
    options.input = defaults.images;
    options.labels = defaults.labels;
    options.output = defaults.output.empty() ? fs::path("output") : fs::path(defaults.output);
    options.device = defaults.device;
    options.width = defaults.width;
    options.height = defaults.height;
    options.batch = defaults.batch;
    options.requests = defaults.requests;
    options.opencv_threads = defaults.opencv_threads;
    options.confidence = defaults.confidence;
    options.inference_precision = defaults.use_fp16 ? ov::element::f16 : ov::element::f32;
    options.throughput_mode = defaults.throughput_mode;
    if (options.width <= 0 || options.height <= 0) throw std::runtime_error("Inference size must be positive");
    if (options.batch <= 0) throw std::runtime_error("Batch size must be positive");
    if (options.requests < 0) throw std::runtime_error("Request count cannot be negative");
    if (options.opencv_threads < 0) throw std::runtime_error("OpenCV thread count cannot be negative");
    if (options.confidence < 0.0f || options.confidence > 1.0f)
        throw std::runtime_error("Confidence must be in [0,1]");
    return options;
}

bool is_image(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp" ||
           extension == ".webp";
}

std::vector<fs::path> collect_images(const fs::path& input) {
    std::vector<fs::path> images;
    if (fs::is_regular_file(input) && is_image(input)) {
        images.push_back(input);
    } else if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (entry.is_regular_file() && is_image(entry.path())) images.push_back(entry.path());
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

LabelTable load_labels(const fs::path& path) {
    LabelTable table;
    if (path.empty()) return table;

    std::ifstream file(path, std::ios::binary);
    if (!file) return table;

    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    if (bytes <= 0) return table;
    file.seekg(0);

    table.blob.resize(static_cast<size_t>(bytes) + 1);
    file.read(table.blob.data(), bytes);
    table.blob[static_cast<size_t>(bytes)] = '\0';
    table.names.reserve(256);

    char* cursor = table.blob.data();
    const char* const end = table.blob.data() + table.blob.size();
    while (cursor < end) {
        while (cursor < end && (*cursor == '\r' || *cursor == '\n' || *cursor == '\0')) ++cursor;
        if (cursor >= end || *cursor == '\0') break;
        char* start = cursor;
        while (cursor < end && *cursor != '\r' && *cursor != '\n' && *cursor != '\0') ++cursor;
        *cursor = '\0';
        table.names.push_back(start);
        ++cursor;
    }
    return table;
}

void configure_host_runtime(int requested_threads) {
    cv::setUseOptimized(true);
    const int hardware_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int threads = requested_threads > 0 ? requested_threads : hardware_threads;
    cv::setNumThreads(threads);
}

cv::Scalar color_for(int id) {
    static const std::array<cv::Scalar, 10> colors = {
        cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
        cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
        cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0)
    };
    return colors[static_cast<size_t>(std::max(0, id)) % colors.size()];
}

cv::Mat nhwc_view(ov::Tensor& tensor, int batch_index, int height, int width) {
    const size_t plane = static_cast<size_t>(height) * width * 3;
    return cv::Mat(height, width, CV_8UC3, tensor.data<uint8_t>() + static_cast<size_t>(batch_index) * plane);
}

void classification_into(const cv::Mat& image, cv::Mat& dst) {
    const float scale = std::max(static_cast<float>(dst.cols) / image.cols, static_cast<float>(dst.rows) / image.rows);
    const int crop_w = std::min(image.cols, std::max(1, cvRound(dst.cols / scale)));
    const int crop_h = std::min(image.rows, std::max(1, cvRound(dst.rows / scale)));
    const int x = std::max(0, (image.cols - crop_w) / 2);
    const int y = std::max(0, (image.rows - crop_h) / 2);
    cv::resize(image(cv::Rect(x, y, crop_w, crop_h)), dst, dst.size(), 0, 0, cv::INTER_LINEAR);
}

void letterbox_into(const cv::Mat& image, cv::Mat& dst, Transform& transform) {
    transform.scale = std::min(static_cast<float>(dst.cols) / image.cols, static_cast<float>(dst.rows) / image.rows);
    transform.inv_scale = 1.0f / transform.scale;
    transform.resized_w = std::min(dst.cols, cvRound(image.cols * transform.scale));
    transform.resized_h = std::min(dst.rows, cvRound(image.rows * transform.scale));
    const float half_pad_x = static_cast<float>(dst.cols - transform.resized_w) * 0.5f;
    const float half_pad_y = static_cast<float>(dst.rows - transform.resized_h) * 0.5f;
    // Ultralytics LetterBox uses round(dh/dw - 0.1) for the top/left edge.
    transform.pad_x = cvRound(half_pad_x - 0.1f);
    transform.pad_y = cvRound(half_pad_y - 0.1f);
    const int bottom = dst.rows - transform.pad_y - transform.resized_h;
    const int right = dst.cols - transform.pad_x - transform.resized_w;
    // Initialize just the four borders, including an asymmetric last row/column.
    if (transform.pad_y > 0) dst.rowRange(0, transform.pad_y).setTo(kLetterboxPad);
    if (bottom > 0) dst.rowRange(dst.rows - bottom, dst.rows).setTo(kLetterboxPad);
    cv::Mat middle = dst.rowRange(transform.pad_y, transform.pad_y + transform.resized_h);
    if (transform.pad_x > 0) middle.colRange(0, transform.pad_x).setTo(kLetterboxPad);
    if (right > 0) middle.colRange(dst.cols - right, dst.cols).setTo(kLetterboxPad);
    cv::resize(image, dst(cv::Rect(transform.pad_x, transform.pad_y, transform.resized_w, transform.resized_h)),
               cv::Size(transform.resized_w, transform.resized_h), 0, 0, cv::INTER_LINEAR);
}

cv::Point2f restore_point(float x, float y, const Transform& transform, const cv::Size& size) {
    return {std::clamp((x - transform.pad_x) * transform.inv_scale, 0.0f, static_cast<float>(size.width)),
            std::clamp((y - transform.pad_y) * transform.inv_scale, 0.0f, static_cast<float>(size.height))};
}

cv::Point2f restore_unclipped_point(float x, float y, const Transform& transform) {
    return {(x - transform.pad_x) * transform.inv_scale, (y - transform.pad_y) * transform.inv_scale};
}

cv::Rect2f restore_box(float x1, float y1, float x2, float y2, const Transform& transform, const cv::Size& size) {
    const cv::Point2f a = restore_point(x1, y1, transform, size);
    const cv::Point2f b = restore_point(x2, y2, transform, size);
    return {a.x, a.y, std::max(0.0f, b.x - a.x), std::max(0.0f, b.y - a.y)};
}

std::vector<Detection> decode_rows(const float* data, size_t rows, size_t columns, Task task,
                                   const Transform& transform, const cv::Size& image_size, float confidence) {
    std::vector<Detection> detections;
    detections.reserve(rows);
    const bool pose = task == Task::Pose && columns >= 57;
    const bool seg = task == Task::InstanceSegmentation && columns > 6;
    const bool obb = task == Task::OrientedBoundingBox;
    for (size_t row_index = 0; row_index < rows; ++row_index) {
        const float* row = data + row_index * columns;
        if (!(row[4] >= confidence)) continue;
        Detection detection;
        detection.score = row[4];
        detection.class_id = std::max(0, static_cast<int>(row[5] + 0.5f));
        if (obb) {
            // Ultralytics scale_boxes(..., xywh=True) does not clip rotated-box centers.
            const cv::Point2f center = restore_unclipped_point(row[0], row[1], transform);
            const float w = row[2] * transform.inv_scale;
            const float h = row[3] * transform.inv_scale;
            detection.box = {center.x - w * 0.5f, center.y - h * 0.5f, w, h};
            detection.angle = row[6];
        } else {
            detection.network_box = {row[0], row[1], std::max(0.0f, row[2] - row[0]),
                                     std::max(0.0f, row[3] - row[1])};
            detection.box = restore_box(row[0], row[1], row[2], row[3], transform, image_size);
        }
        if (pose) {
            for (int k = 0; k < 17; ++k) {
                detection.keypoints[k] = restore_point(row[6 + k * 3], row[7 + k * 3], transform, image_size);
                detection.keypoint_scores[k] = row[8 + k * 3];
            }
        } else if (seg) {
            detection.mask_coefficient_count = std::min(columns - 6, detection.mask_coefficients.size());
            std::copy_n(row + 6, detection.mask_coefficient_count, detection.mask_coefficients.begin());
        }
        detections.push_back(std::move(detection));
    }
    return detections;
}

void draw_box(cv::Mat& image, const Detection& detection, Task task, const LabelTable& labels) {
    const cv::Scalar color = color_for(detection.class_id);
    if (task == Task::OrientedBoundingBox) {
        cv::RotatedRect rotated({detection.box.x + detection.box.width * 0.5f, detection.box.y + detection.box.height * 0.5f},
                                detection.box.size(), detection.angle * 180.0f / static_cast<float>(CV_PI));
        cv::Point2f corners[4];
        rotated.points(corners);
        for (int i = 0; i < 4; ++i) cv::line(image, corners[i], corners[(i + 1) % 4], color, 2, cv::LINE_AA);
    } else {
        cv::rectangle(image, detection.box, color, 2, cv::LINE_AA);
    }
    char label[64];
    std::snprintf(label, sizeof(label), "%s %.2f", labels[detection.class_id], detection.score);
    const int x = std::max(0, cvRound(detection.box.x));
    const int y = std::max(16, cvRound(detection.box.y));
    cv::putText(image, label, {x, y - 3}, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

void draw_pose(cv::Mat& image, const Detection& detection) {
    static const std::array<std::pair<int, int>, 16> skeleton = {{{0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8},
                                                                   {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}}};
    for (const auto& edge : skeleton) {
        if (detection.keypoint_scores[edge.first] > 0.5f && detection.keypoint_scores[edge.second] > 0.5f)
            cv::line(image, detection.keypoints[edge.first], detection.keypoints[edge.second], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    for (int k = 0; k < 17; ++k) {
        if (detection.keypoint_scores[k] > 0.5f) cv::circle(image, detection.keypoints[k], 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
}

cv::Rect scaled_valid_roi(const Transform& transform, int source_w, int source_h, int target_w, int target_h) {
    const float sx = static_cast<float>(target_w) / source_w;
    const float sy = static_cast<float>(target_h) / source_h;
    int x = std::clamp(cvRound(transform.pad_x * sx), 0, target_w - 1);
    int y = std::clamp(cvRound(transform.pad_y * sy), 0, target_h - 1);
    int w = std::clamp(cvRound(transform.resized_w * sx), 1, target_w - x);
    int h = std::clamp(cvRound(transform.resized_h * sy), 1, target_h - y);
    return {x, y, w, h};
}

void overlay_instance_masks(cv::Mat& image, const std::vector<Detection>& detections, const ov::Tensor& prototypes,
                            const Transform& transform, int network_w, int network_h, size_t batch_index) {
    const ov::Shape shape = prototypes.get_shape();
    if (shape.size() != 4 || shape[0] == 0 || shape[1] == 0) throw std::runtime_error("Unexpected prototype tensor shape");
    if (batch_index >= shape[0]) throw std::runtime_error("Prototype batch index out of range");
    const int channels = static_cast<int>(shape[1]);
    const int proto_h = static_cast<int>(shape[2]);
    const int proto_w = static_cast<int>(shape[3]);
    const float* proto = prototypes.data<const float>() +
                         batch_index * static_cast<size_t>(channels) * proto_h * proto_w;
    cv::Mat proto_mat(channels, proto_h * proto_w, CV_32F, const_cast<float*>(proto));
    const cv::Rect valid_roi = scaled_valid_roi(transform, network_w, network_h, network_w, network_h);
    cv::Mat logits;
    cv::Mat network_logits;
    cv::Mat network_mask;
    cv::Mat valid_float;
    cv::Mat restored;
    cv::Mat tinted;
    for (const auto& detection : detections) {
        if (static_cast<int>(detection.mask_coefficient_count) != channels) continue;

        cv::Mat coeffs(1, channels, CV_32F, const_cast<float*>(detection.mask_coefficients.data()));
        cv::gemm(coeffs, proto_mat, 1.0, cv::noArray(), 0.0, logits);
        cv::Mat logits_hw(proto_h, proto_w, CV_32F, logits.ptr<float>());
        // Match Ultralytics process_mask: upsample logits first, threshold at zero,
        // then crop with the floating-point network-space box.
        cv::resize(logits_hw, network_logits, cv::Size(network_w, network_h), 0, 0, cv::INTER_LINEAR);
        network_mask = network_logits > 0.0f;

        const float x1 = detection.network_box.x;
        const float y1 = detection.network_box.y;
        const float x2 = x1 + detection.network_box.width;
        const float y2 = y1 + detection.network_box.height;
        const int left = std::clamp(static_cast<int>(std::ceil(x1)), 0, network_w);
        const int top = std::clamp(static_cast<int>(std::ceil(y1)), 0, network_h);
        const int right = std::clamp(static_cast<int>(std::ceil(x2)), 0, network_w);
        const int bottom = std::clamp(static_cast<int>(std::ceil(y2)), 0, network_h);
        if (right <= left || bottom <= top) continue;
        if (top > 0) network_mask.rowRange(0, top).setTo(0);
        if (bottom < network_h) network_mask.rowRange(bottom, network_h).setTo(0);
        if (left > 0) network_mask.colRange(0, left).setTo(0);
        if (right < network_w) network_mask.colRange(right, network_w).setTo(0);

        // Ultralytics plotting removes LetterBox padding, bilinear-resizes the binary
        // input-resolution mask, and thresholds it again at 0.5.
        network_mask(valid_roi).convertTo(valid_float, CV_32F, 1.0 / 255.0);
        cv::resize(valid_float, restored, image.size(), 0, 0, cv::INTER_LINEAR);
        restored = restored > 0.5f;
        if (cv::countNonZero(restored) == 0) continue;
        image.copyTo(tinted);
        tinted.setTo(color_for(detection.class_id), restored);
        cv::addWeighted(tinted, 0.45, image, 0.55, 0, image);
    }
}

void save_classification(cv::Mat& image, const ov::Tensor& output, const LabelTable& labels, const fs::path& path,
                         size_t batch_index) {
    const ov::Shape shape = output.get_shape();
    const size_t count = shape.empty() ? 0 : shape.back();
    const size_t offset = shape.size() >= 2 ? batch_index * count : 0;
    const float* scores = output.data<const float>() + offset;
    const size_t top_count = std::min<size_t>(5, count);
    std::array<size_t, 5> indices{};
    std::array<float, 5> top_scores{};
    top_scores.fill(-std::numeric_limits<float>::infinity());
    for (size_t id = 0; id < count; ++id) {
        size_t position = top_count;
        for (size_t rank = 0; rank < top_count; ++rank) {
            if (scores[id] > top_scores[rank]) {
                position = rank;
                break;
            }
        }
        if (position == top_count) continue;
        for (size_t rank = top_count - 1; rank > position; --rank) {
            top_scores[rank] = top_scores[rank - 1];
            indices[rank] = indices[rank - 1];
        }
        top_scores[position] = scores[id];
        indices[position] = id;
    }
    const int bar_h = std::min(image.rows, 35 + static_cast<int>(top_count) * 27);
    cv::Mat bar = image.rowRange(0, bar_h);
    cv::Mat overlay = bar.clone();
    overlay.setTo(cv::Scalar(20, 20, 20));
    cv::addWeighted(overlay, 0.65, bar, 0.35, 0, bar);
    for (size_t rank = 0; rank < top_count; ++rank) {
        const size_t id = indices[rank];
        char text[160];
        std::snprintf(text, sizeof(text), "%zu. %s  %.3f", rank + 1, labels[static_cast<int>(id)], scores[id]);
        cv::putText(image, text, {10, 28 + static_cast<int>(rank) * 27}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    rank == 0 ? cv::Scalar(80, 230, 80) : cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        std::cout << "    " << text << '\n';
    }
    cv::imwrite(path.string(), image, {cv::IMWRITE_JPEG_QUALITY, 90});
}

void save_depth(const ov::Tensor& output, const Transform& transform, int network_w, int network_h,
                const cv::Size& image_size, const fs::path& image_path, const fs::path& binary_path, size_t batch_index) {
    const ov::Shape shape = output.get_shape();
    if (shape.size() != 4 || shape[1] != 1) throw std::runtime_error("Depth output must be [N,1,H,W]");
    if (batch_index >= shape[0]) throw std::runtime_error("Depth batch index out of range");
    const int height = static_cast<int>(shape[2]);
    const int width = static_cast<int>(shape[3]);
    cv::Mat depth(height, width, CV_32F,
                  const_cast<float*>(output.data<const float>() + batch_index * static_cast<size_t>(height) * width));
    const cv::Rect valid_roi = scaled_valid_roi(transform, network_w, network_h, width, height);
    cv::Mat restored;
    cv::resize(depth(valid_roi), restored, image_size, 0, 0, cv::INTER_LINEAR);
    std::ofstream binary(binary_path, std::ios::binary);
    binary.write(reinterpret_cast<const char*>(restored.ptr<float>()),
                 static_cast<std::streamsize>(restored.total() * sizeof(float)));
    double minimum = 0.0, maximum = 0.0;
    cv::minMaxLoc(restored, &minimum, &maximum);
    cv::Mat normalized, colored;
    restored.convertTo(normalized, CV_8U, 255.0 / std::max(1e-12, maximum - minimum),
                       -minimum * 255.0 / std::max(1e-12, maximum - minimum));
    cv::applyColorMap(normalized, colored, cv::COLORMAP_INFERNO);
    cv::imwrite(image_path.string(), colored);
}

void blend_semantic(cv::Mat& image, const cv::Mat& class_map) {
    cv::parallel_for_(cv::Range(0, image.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            cv::Vec3b* dst = image.ptr<cv::Vec3b>(y);
            const uint8_t* cls = class_map.ptr<uint8_t>(y);
            for (int x = 0; x < image.cols; ++x) {
                const cv::Vec3b color = kCityLut[cls[x]];
                dst[x][0] = static_cast<uint8_t>((dst[x][0] * 115 + color[0] * 141) >> 8);
                dst[x][1] = static_cast<uint8_t>((dst[x][1] * 115 + color[1] * 141) >> 8);
                dst[x][2] = static_cast<uint8_t>((dst[x][2] * 115 + color[2] * 141) >> 8);
            }
        }
    });
}

void argmax_nchw(const float* logits, int classes, int height, int width, cv::Mat& class_map) {
    class_map.create(height, width, CV_8U);
    const int hw = height * width;
    uint8_t* dst = class_map.ptr<uint8_t>();
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
        std::vector<float> best(static_cast<size_t>(width));
        for (int y = range.start; y < range.end; ++y) {
            const float* plane0 = logits + y * width;
            uint8_t* row = dst + y * width;
            std::copy(plane0, plane0 + width, best.begin());
            std::fill(row, row + width, static_cast<uint8_t>(0));
            for (int c = 1; c < classes; ++c) {
                const float* plane = logits + c * hw + y * width;
                for (int x = 0; x < width; ++x) {
                    if (plane[x] > best[static_cast<size_t>(x)]) {
                        best[static_cast<size_t>(x)] = plane[x];
                        row[x] = static_cast<uint8_t>(c);
                    }
                }
            }
        }
    });
}

void save_semantic(cv::Mat& image, const ov::Tensor& output, const Transform& transform, int network_w, int network_h,
                   const fs::path& result_path, const fs::path& mask_path, size_t batch_index) {
    const ov::Shape shape = output.get_shape();
    cv::Mat class_map;
    if (shape.size() == 3 && output.get_element_type() == ov::element::u8) {
        if (batch_index >= shape[0]) throw std::runtime_error("Semantic batch index out of range");
        const int height = static_cast<int>(shape[1]);
        const int width = static_cast<int>(shape[2]);
        class_map = cv::Mat(height, width, CV_8U,
                            const_cast<uint8_t*>(output.data<const uint8_t>() +
                                                 batch_index * static_cast<size_t>(height) * width));
    } else if (shape.size() == 4 && output.get_element_type() == ov::element::f32) {
        if (batch_index >= shape[0]) throw std::runtime_error("Semantic batch index out of range");
        const int classes = static_cast<int>(shape[1]);
        const int height = static_cast<int>(shape[2]);
        const int width = static_cast<int>(shape[3]);
        const float* logits = output.data<const float>() +
                              batch_index * static_cast<size_t>(classes) * height * width;
        argmax_nchw(logits, classes, height, width, class_map);
    } else {
        throw std::runtime_error("Semantic output must be uint8 [N,H,W] or float32 [N,C,H,W]");
    }
    const cv::Rect valid_roi = scaled_valid_roi(transform, network_w, network_h, class_map.cols, class_map.rows);
    cv::Mat restored;
    cv::resize(class_map(valid_roi), restored, image.size(), 0, 0, cv::INTER_NEAREST);
    blend_semantic(image, restored);
    cv::imwrite(result_path.string(), image, {cv::IMWRITE_JPEG_QUALITY, 90});
    cv::imwrite(mask_path.string(), restored);
}

std::string shape_string(const ov::Shape& shape) {
    std::ostringstream stream;
    stream << '[';
    for (size_t i = 0; i < shape.size(); ++i) stream << (i ? "," : "") << shape[i];
    stream << ']';
    return stream.str();
}

std::shared_ptr<ov::Model> prepare_model(std::shared_ptr<ov::Model> model, int batch, int height, int width) {
    if (model->inputs().size() != 1) throw std::runtime_error("Expected exactly one model input");
    const std::string input_name = model->input().get_any_name();
    std::cout << "IR input " << input_name << ' ' << model->input().get_partial_shape() << " -> "
              << '[' << batch << ",3," << height << ',' << width << ']' << std::endl;
    model->reshape({{input_name, ov::PartialShape{batch, 3, height, width}}});

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR)
        .set_shape({static_cast<size_t>(batch), static_cast<size_t>(height), static_cast<size_t>(width), 3});
    ppp.input().model().set_layout("NCHW");
    ppp.input()
        .preprocess()
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .convert_element_type(ov::element::f32)
        .scale(255.0f);
    return ppp.build();
}

ov::CompiledModel compile_for_device(ov::Core& core, const std::shared_ptr<ov::Model>& model,
                                     const Options& options) {
    const auto t0 = std::chrono::steady_clock::now();
    ov::AnyMap properties;
    properties[ov::hint::performance_mode.name()] = options.throughput_mode
                                                       ? ov::hint::PerformanceMode::THROUGHPUT
                                                       : ov::hint::PerformanceMode::LATENCY;
    properties[ov::hint::inference_precision.name()] = options.inference_precision;
    properties[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    if (options.requests > 0)
        properties[ov::hint::num_requests.name()] = static_cast<unsigned int>(options.requests);
    properties[ov::hint::allow_auto_batching.name()] = false;
    if (options.device == "CPU" || options.device == "AUTO") {
        properties[ov::intel_cpu::denormals_optimization.name()] = true;
        // Pinning/reservation can deadlock compile_model on Windows when the VS
        // debugger or OpenCV has already created a TBB arena.
        properties[ov::hint::enable_cpu_pinning.name()] = false;
        properties[ov::hint::enable_cpu_reservation.name()] = false;
    }

    ov::CompiledModel compiled = core.compile_model(model, options.device, properties);
    const double compile_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "Compile finished in " << std::fixed << std::setprecision(0) << compile_ms << " ms" << std::endl;
    return compiled;
}

void save_detections(cv::Mat& image, const std::vector<Detection>& detections, Task task, const ov::CompiledModel& compiled,
                     ov::InferRequest& request, const Transform& transform, int width, int height, size_t batch_index,
                     const fs::path& path, const LabelTable& labels) {
    if (task == Task::InstanceSegmentation) {
        if (compiled.outputs().size() != 2) throw std::runtime_error("Instance segmentation model needs two outputs");
        overlay_instance_masks(image, detections, request.get_output_tensor(1), transform, width, height, batch_index);
    }
    for (const auto& detection : detections) {
        draw_box(image, detection, task, labels);
        if (task == Task::Pose) draw_pose(image, detection);
    }
    cv::imwrite(path.string(), image, {cv::IMWRITE_JPEG_QUALITY, 90});
}

struct RequestSlot {
    RequestSlot(ov::CompiledModel& compiled, size_t batch)
        : request(compiled.create_infer_request()),
          input(request.get_input_tensor()),
          originals(batch),
          transforms(batch),
          paths(batch) {}

    ov::InferRequest request;
    ov::Tensor input;
    std::vector<cv::Mat> originals;
    std::vector<Transform> transforms;
    std::vector<fs::path> paths;
    std::chrono::steady_clock::time_point submitted_at{};
    size_t start_index = 0;
    size_t actual = 0;
    bool busy = false;
};

void prepare_batch(RequestSlot& slot, const std::vector<fs::path>& images, size_t start, int batch,
                   int width, int height, Task task) {
    slot.start_index = start;
    slot.actual = std::min(static_cast<size_t>(batch), images.size() - start);
    for (size_t i = 0; i < slot.actual; ++i) {
        slot.originals[i].release();
        slot.transforms[i] = Transform{};
        slot.paths[i] = images[start + i];
    }

    auto preprocess_one = [&](int i) {
        cv::Mat image = cv::imread(slot.paths[static_cast<size_t>(i)].string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Cannot read image: " + slot.paths[static_cast<size_t>(i)].string());
        cv::Mat view = nhwc_view(slot.input, i, height, width);
        Transform transform;
        if (task == Task::Classification) classification_into(image, view);
        else letterbox_into(image, view, transform);
        slot.originals[static_cast<size_t>(i)] = std::move(image);
        slot.transforms[static_cast<size_t>(i)] = transform;
    };

    if (slot.actual > 1) {
        cv::parallel_for_(cv::Range(0, static_cast<int>(slot.actual)), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) preprocess_one(i);
        });
    } else {
        preprocess_one(0);
    }

    // A compiled model has a fixed runtime N. Pad only the final batch by copying
    // the already-preprocessed last item; no extra decode or resize is performed.
    if (slot.actual < static_cast<size_t>(batch)) {
        cv::Mat last = nhwc_view(slot.input, static_cast<int>(slot.actual - 1), height, width);
        for (size_t i = slot.actual; i < static_cast<size_t>(batch); ++i) {
            last.copyTo(nhwc_view(slot.input, static_cast<int>(i), height, width));
        }
    }
}

double finish_batch(RequestSlot& slot, const std::vector<fs::path>& images, Task task, const Options& options,
                    const LabelTable& labels, const ov::CompiledModel& compiled) {
    slot.request.wait();
    const double request_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - slot.submitted_at)
                                  .count();
    const ov::Tensor output0 = slot.request.get_output_tensor(0);

    for (size_t i = 0; i < slot.actual; ++i) {
        std::cout << '[' << (slot.start_index + i + 1) << '/' << images.size() << "] "
                  << slot.paths[i].filename().string() << "  " << std::fixed << std::setprecision(2)
                  << request_ms << " ms (batch=" << options.batch << ")\n";
        const std::string stem = slot.paths[i].stem().string();
        if (task == Task::Classification) {
            save_classification(slot.originals[i], output0, labels, options.output / (stem + "_cls.jpg"), i);
            continue;
        }
        if (task == Task::Depth) {
            save_depth(output0, slot.transforms[i], options.width, options.height, slot.originals[i].size(),
                       options.output / (stem + "_depth.png"), options.output / (stem + "_depth.bin"), i);
            continue;
        }
        if (task == Task::SemanticSegmentation) {
            save_semantic(slot.originals[i], output0, slot.transforms[i], options.width, options.height,
                          options.output / (stem + "_sem.jpg"), options.output / (stem + "_mask.png"), i);
            continue;
        }

        const ov::Shape rows_shape = output0.get_shape();
        if (rows_shape.size() != 3) throw std::runtime_error("Detection output must be [N,K,C]");
        if (i >= rows_shape[0]) throw std::runtime_error("Detection batch index out of range");
        auto detections = decode_rows(output0.data<const float>() + i * rows_shape[1] * rows_shape[2],
                                      rows_shape[1], rows_shape[2], task, slot.transforms[i],
                                      slot.originals[i].size(), options.confidence);
        save_detections(slot.originals[i], detections, task, compiled, slot.request, slot.transforms[i],
                        options.width, options.height, i, options.output / (stem + "_result.jpg"), labels);
        std::cout << "    detections: " << detections.size() << '\n';
    }
    slot.busy = false;
    return request_ms;
}

}  // namespace

int run(Task task, const Defaults& defaults) {
    try {
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
        Options options = make_options(defaults);
        const auto images = collect_images(options.input);
        if (images.empty()) throw std::runtime_error("No images found at: " + options.input.string());
        if (!fs::exists(options.model)) throw std::runtime_error("Model not found: " + options.model.string());
        fs::create_directories(options.output);

        const size_t batch_count = (images.size() + static_cast<size_t>(options.batch) - 1) /
                                   static_cast<size_t>(options.batch);
        cv::setUseOptimized(true);
        cv::setNumThreads(1);

        const LabelTable labels = load_labels(options.labels);

        const int batch = options.batch;
        std::cout << ov::get_openvino_version() << "\n"
                  << "YOLO26 OpenVINO " << task_name(task) << "\n"
                  << "Model:  " << options.model << "\nDevice: " << options.device
                  << "\nInput:  " << options.width << 'x' << options.height
                  << "  batch " << batch << ", requests "
                  << (options.requests == 0 ? std::string("auto") : std::to_string(options.requests))
                  << " (dynamic IR reshape, reusable u8 NHWC tensors, async/"
                  << options.inference_precision.get_type_name() << ')' << std::endl;
        if (!options.labels.empty()) {
            if (labels.empty()) {
                std::cout << "Labels: " << options.labels << " (missing or empty, fallback class_id)" << std::endl;
            } else {
                std::cout << "Labels: " << options.labels << " (" << labels.size() << " classes, in-memory)" << std::endl;
            }
        }

        ov::Core core;
        // mmap + VS debugger first-chance AVs inside CPU JIT (unknown addresses, no PDB).
        core.set_property(ov::enable_mmap(false));
        // models/best.xml -> <project>/openvino_cache/{Debug|Release}
        const fs::path project_root = fs::absolute(options.model).parent_path().parent_path();
#ifdef _DEBUG
        const fs::path cache_dir = project_root / "openvino_cache" / "Debug";
#else
        const fs::path cache_dir = project_root / "openvino_cache" / "Release";
#endif
        fs::create_directories(cache_dir);
        core.set_property(ov::cache_dir(cache_dir.string()));

        std::cout << "Loading IR..." << std::endl;
        std::shared_ptr<ov::Model> model = prepare_model(core.read_model(options.model.string()), batch,
                                                         options.height, options.width);

        std::cout << "Compiling on " << options.device << " (cache " << cache_dir.string()
                  << "; first compile of this shape can take 1-3 min)..." << std::endl;
        ov::CompiledModel compiled = compile_for_device(core, model, options);
        if (options.requests == 0) {
            options.requests = static_cast<int>(compiled.get_property(ov::optimal_number_of_infer_requests));
        }
        options.requests = std::clamp(options.requests, 1, static_cast<int>(batch_count));
        configure_host_runtime(options.opencv_threads);
        std::vector<RequestSlot> slots;
        slots.reserve(static_cast<size_t>(options.requests));
        for (int i = 0; i < options.requests; ++i) slots.emplace_back(compiled, static_cast<size_t>(batch));
        if (slots.front().input.get_element_type() != ov::element::u8)
            throw std::runtime_error("Expected u8 NHWC input after PrePostProcessor");
        std::cout << "Compiled input " << compiled.input().get_any_name() << ' '
                  << slots.front().input.get_element_type() << ' ' << shape_string(slots.front().input.get_shape())
                  << ", infer requests " << options.requests
                  << ", OpenCV threads " << cv::getNumThreads() << std::endl;

        // Warm up kernels and memory paths without allocating a temporary input tensor.
        std::fill_n(slots.front().input.data<uint8_t>(), slots.front().input.get_size(), uint8_t{114});
        slots.front().request.infer();
        slots.front().request.infer();

        double request_ms_sum = 0.0;
        size_t completed_images = 0;
        size_t submission = 0;
        const auto pipeline_start = std::chrono::steady_clock::now();
        for (size_t start = 0; start < images.size(); start += static_cast<size_t>(batch)) {
            RequestSlot& slot = slots[submission % slots.size()];
            if (slot.busy) {
                request_ms_sum += finish_batch(slot, images, task, options, labels, compiled);
                completed_images += slot.actual;
            }
            prepare_batch(slot, images, start, batch, options.width, options.height, task);
            slot.submitted_at = std::chrono::steady_clock::now();
            slot.busy = true;
            slot.request.start_async();
            ++submission;
        }

        std::vector<RequestSlot*> pending;
        pending.reserve(slots.size());
        for (RequestSlot& slot : slots) if (slot.busy) pending.push_back(&slot);
        std::sort(pending.begin(), pending.end(), [](const RequestSlot* a, const RequestSlot* b) {
            return a->start_index < b->start_index;
        });
        for (RequestSlot* slot : pending) {
            request_ms_sum += finish_batch(*slot, images, task, options, labels, compiled);
            completed_images += slot->actual;
        }
        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pipeline_start).count();
        std::cout << "Done. Average request latency: " << std::fixed << std::setprecision(2)
                  << request_ms_sum / std::max<size_t>(completed_images, 1)
                  << " ms/image, e2e: " << wall_ms / images.size() << " ms/image, throughput: "
                  << (1000.0 * images.size() / std::max(wall_ms, 1e-6)) << " fps, results: "
                  << fs::absolute(options.output) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[Error] " << error.what() << '\n';
        return 1;
    }
}

}  // namespace yolo26
