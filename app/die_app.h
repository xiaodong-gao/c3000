#pragma once

#include "die_types.h"

namespace cutdie {

std::vector<Job> collect_jobs(const Options& opt);

// 配对文件，按 XML 切割 patch，再用外部模板做完整检出。失败返回非 0。
int run(Options opt);

}  // namespace cutdie
