#include "die_cli.h"

#include "die_config.h"
#include "die_util.h"

#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace cutdie {
namespace {

const std::string& require_value(const std::vector<std::string>& args, size_t& index,
                                 const std::string& option) {
    if (++index >= args.size()) {
        throw std::invalid_argument(option + " 缺少参数");
    }
    return args[index];
}

int parse_positive_int(const std::string& text, const std::string& option) {
    size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " 需要正整数: " + text);
    }
    if (used != text.size() || value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(option + " 需要正整数: " + text);
    }
    return static_cast<int>(value);
}

double parse_positive_double(const std::string& text, const std::string& option) {
    size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " 需要正数: " + text);
    }
    if (used != text.size() || !std::isfinite(value) || !(value > 0.0)) {
        throw std::invalid_argument(option + " 需要正数: " + text);
    }
    return value;
}

void select_mode(CommandLine& command, RunMode mode, const std::string& option) {
    if (command.mode != RunMode::Process && command.mode != mode) {
        throw std::invalid_argument(option + " 与其他运行模式冲突");
    }
    command.mode = mode;
}

AlignMethod parse_align_method(const std::string& text) {
    const std::string value = to_lower(text);
    if (value == "phase-cpu") return AlignMethod::PhaseCpu;
    if (value == "phase-cuda") return AlignMethod::PhaseCuda;
    if (value == "ecc-cpu") return AlignMethod::EccCpu;
    if (value == "ecc-cuda") return AlignMethod::EccCuda;
    throw std::invalid_argument("未知 --align 值: " + text);
}

HistogramMethod parse_histogram_method(const std::string& text) {
    const std::string value = to_lower(text);
    if (value == "auto") return HistogramMethod::Auto;
    if (value == "cpu") return HistogramMethod::Cpu;
    if (value == "cuda") return HistogramMethod::Cuda;
    throw std::invalid_argument("未知 --histogram 值: " + text);
}

NormalizeMethod parse_normalize_method(const std::string& text) {
    const std::string value = to_lower(text);
    if (value == "auto") return NormalizeMethod::Auto;
    if (value == "cpu") return NormalizeMethod::Cpu;
    if (value == "cuda") return NormalizeMethod::Cuda;
    throw std::invalid_argument("未知 --normalize 值: " + text);
}

}  // namespace

CommandLine parse_command_line(const std::vector<std::string>& args) {
    CommandLine command;
    command.options = make_default_options();
    if (args.empty()) {
        command.mode = RunMode::Help;
        return command;
    }

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--help" || arg == "-h") {
            select_mode(command, RunMode::Help, arg);
        } else if (arg == "--self-test" || arg == "--test") {
            select_mode(command, RunMode::Regression, arg);
        } else if (arg == "--benchmark") {
            select_mode(command, RunMode::Benchmark, arg);
            if (i + 1 < args.size() && !args[i + 1].empty() && args[i + 1][0] != '-') {
                command.benchmark_iterations = parse_positive_int(args[++i], arg);
            }
        } else if (arg == "--benchmark-iters") {
            select_mode(command, RunMode::Benchmark, arg);
            command.benchmark_iterations =
                parse_positive_int(require_value(args, i, arg), arg);
        } else if (arg == "--image") {
            command.options.image = require_value(args, i, arg);
        } else if (arg == "--xml") {
            command.options.xml = require_value(args, i, arg);
        } else if (arg == "--dir") {
            command.options.dir = require_value(args, i, arg);
            command.options.image.clear();
            command.options.xml.clear();
        } else if (arg == "--out") {
            command.options.out = require_value(args, i, arg);
        } else if (arg == "--patch-out") {
            command.options.patch_out = require_value(args, i, arg);
        } else if (arg == "--template") {
            command.options.template_path = require_value(args, i, arg);
        } else if (arg == "--eval-patches") {
            command.options.eval_patch_dir = require_value(args, i, arg);
        } else if (arg == "--eval-truth") {
            command.options.eval_truth_dir = require_value(args, i, arg);
        } else if (arg == "--resolution-um") {
            command.options.resolution_um =
                parse_positive_double(require_value(args, i, arg), arg);
        } else if (arg == "--align") {
            command.options.align_method = parse_align_method(require_value(args, i, arg));
        } else if (arg == "--histogram") {
            command.options.histogram_method =
                parse_histogram_method(require_value(args, i, arg));
        } else if (arg == "--normalize") {
            command.options.normalize_method =
                parse_normalize_method(require_value(args, i, arg));
        } else if (arg == "--template-neighbor") {
            command.options.template_neighbor = true;
            command.options.template_neighbor_min =
                parse_positive_int(require_value(args, i, arg), arg);
        } else if (arg == "--no-template-neighbor") {
            command.options.template_neighbor = false;
        } else if (arg == "--recursive") {
            command.options.recursive = true;
        } else if (arg == "--keep-skipped") {
            command.options.keep_skipped = true;
        } else if (arg == "--inspect") {
            command.options.inspect = true;
        } else if (arg == "--no-inspect") {
            command.options.inspect = false;
        } else {
            throw std::invalid_argument("未知参数: " + arg);
        }
    }
    return command;
}

void print_usage(std::ostream& out) {
    out <<
        "c3000 - wafer die cutting and D2D inspection\n\n"
        "Usage:\n"
        "  c3000 --self-test\n"
        "  c3000 --benchmark [iterations]\n"
        "  c3000 [processing options]\n\n"
        "Processing options:\n"
        "  --image <png>                 process one wafer image\n"
        "  --xml <file>                  die-map XML for --image\n"
        "  --dir <dir>                   scan a directory (clears default image/XML)\n"
        "  --out <dir>                   output root\n"
        "  --patch-out <dir>             patch output directory\n"
        "  --template <image-or-dir>     external template source\n"
        "  --template-neighbor <count>   enable neighbor templates and vote count\n"
        "  --no-template-neighbor        disable neighbor templates\n"
        "  --resolution-um <value>       micrometers per pixel\n"
        "  --align <phase-cpu|phase-cuda|ecc-cpu|ecc-cuda>\n"
        "  --histogram <auto|cpu|cuda>\n"
        "  --normalize <auto|cpu|cuda>\n"
        "  --recursive | --keep-skipped | --no-inspect\n"
        "  --eval-patches <dir> --eval-truth <dir> --template <source>\n\n"
        "Algorithm defaults are defined in src/die_config.cpp; input paths must be supplied.\n";
}

}  // namespace cutdie
