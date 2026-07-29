#include <iostream>
#include <thread>

#include "config.hpp"
#include "performance_monitor.hpp"

int main()
{
    std::cout << "=================================\n";
    std::cout << "Driver Monitoring System\n";
    std::cout << "Modern C++ Reference Implementation\n";
    std::cout << "Version: 2.0\n";
    std::cout << "=================================\n";

    dms::PerformanceMonitor perf;

    perf.start("capture");

    std::this_thread::sleep_for(std::chrono::milliseconds(25));

    perf.stop("capture");

    perf.displayPerformance();

    return 0;
}