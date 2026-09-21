#pragma once

#include "die_types.h"

namespace cutdie {

std::vector<Job> collect_jobs(const Options& opt);

// 配对文件，按 XML 切割并保存 patch。失败返回非 0。
int run(Options opt);

}  // namespace cutdie
