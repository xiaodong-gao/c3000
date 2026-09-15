#pragma once

#include "die_inspect.h"

namespace cutdie {

// 只截取对齐 XML 中的圆形 ROI，把该图直接送入 singledie 检出。不切 patch。
InspectResult inspect_singledie(const Options& opt);

}  // namespace cutdie
