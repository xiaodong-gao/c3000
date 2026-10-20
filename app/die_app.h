#pragma once

#include "die_types.h"

namespace cutdie {

std::vector<Job> collect_jobs(const Options& opt);

// 按 Options.mode 调度：各模式都先切并保存全部 die patch。
// DieToDie 用相邻 die 比较；DieToDatabase 用一张参考 patch 比较；
// SingleDie 再对圆形 ROI 做本片检出；YoloOpenVino 再做语义分割。
int run(Options opt);

}  // namespace cutdie
