#pragma once

#include "die_types.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace cutdie {

enum class RunMode { Process, Regression, Benchmark, Help };

struct CommandLine {
    RunMode mode = RunMode::Process;
    Options options;
    int benchmark_iterations = 8;
};

// Parses arguments excluding argv[0]. Throws std::invalid_argument on malformed input.
CommandLine parse_command_line(const std::vector<std::string>& args);

void print_usage(std::ostream& out);

}  // namespace cutdie
