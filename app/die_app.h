#pragma once

#include "die_types.h"

namespace cutdie {

std::vector<Job> collect_jobs(const Options& opt);

// 按 Options.mode 调度：DieToDie 切 patch 做相邻比较；SingleDie 切圆形 ROI 做本片检出。
int run(Options opt);

}  // namespace cutdie
