#pragma once

#include "die_inspect.h"

namespace cutdie {

// 读取一张参考模板，和本片切下来的每个 die patch 做对比。
// 对位和缺陷检测在 die2database 内单独实现，不调用 Die-to-Die 流水线。
InspectResult inspect_die_to_database(const Options& opt, const CutResult& cut);

}  // namespace cutdie
