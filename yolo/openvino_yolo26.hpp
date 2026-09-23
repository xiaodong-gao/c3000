#pragma once

#include <string>

namespace yolo26 {

enum class Task {
    Classification,
    Depth,
    Detection,
    OrientedBoundingBox,
    Pose,
    InstanceSegmentation,
    SemanticSegmentation
};

struct Defaults {
    std::string model;
    std::string images;
    std::string labels;
    std::string output;
    std::string device = "CPU";
    int width = 640;
    int height = 640;
    int batch = 1;
    int requests = 0;
    int opencv_threads = 0;
    float confidence = 0.25f;
    bool use_fp16 = true;
    bool throughput_mode = false;
};

int run(Task task, const Defaults& defaults);

}  // namespace yolo26
