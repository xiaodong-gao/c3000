#include "die_app.h"
#include "die_cli.h"
#include "SelfCheck.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

void configure_console_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

}  // namespace

int main(int argc, char* argv[]) {
    configure_console_utf8();
    try {
        std::vector<std::string> args;
        args.reserve(argc > 1 ? static_cast<size_t>(argc - 1) : 0);
        for (int i = 1; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }

        const cutdie::CommandLine command = cutdie::parse_command_line(args);
        switch (command.mode) {
            case cutdie::RunMode::Help:
                cutdie::print_usage(std::cout);
                return 0;
            case cutdie::RunMode::Regression:
                return runRegressionTests();
            case cutdie::RunMode::Benchmark:
                return runBenchmark(command.benchmark_iterations);
            case cutdie::RunMode::Process:
                return cutdie::run(command.options);
        }
    } catch (const std::invalid_argument& ex) {
        std::cerr << "参数错误: " << ex.what() << "\n\n";
        cutdie::print_usage(std::cerr);
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << '\n';
        return 1;
    }
    return 1;
}
