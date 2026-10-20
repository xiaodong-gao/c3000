#pragma once

#include "die_inspect.h"

namespace cutdie {

// 截取对齐 XML 中的圆形 ROI，把该图直接送入 singledie 检出。
// die patch 由调度侧用 cut_patches 保存，与 Die-to-Die 相同。
InspectResult inspect_singledie(const Options& opt);

}  // namespace cutdie
