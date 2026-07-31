#include "performance_monitor.hpp"

#include <iostream>
#include <iomanip>

namespace dms
{

PerformanceMonitor::PerformanceMonitor()
{
    displayTimer = Clock::now();
}

void PerformanceMonitor::start(const std::string& stage)
{
    startTimes[stage] = Clock::now();
}

void PerformanceMonitor::stop(const std::string& stage)
{
    auto it = startTimes.find(stage);

    if (it != startTimes.end())
    {
        auto elapsed =
            std::chrono::duration<double, std::milli>(
                Clock::now() - it->second).count();

        elapsedTimes[stage] = elapsed;
    }
}

double PerformanceMonitor::get(const std::string& stage) const
{
    auto it = elapsedTimes.find(stage);

    if (it != elapsedTimes.end())
        return it->second;

    return 0.0;
}

double PerformanceMonitor::total() const
{
    double sum = 0.0;

    for (const auto& item : elapsedTimes)
        sum += item.second;

    return sum;
}

void PerformanceMonitor::reset()
{
    elapsedTimes.clear();
}

void PerformanceMonitor::displayPerformance()
{
    auto currentTime = Clock::now();

    auto elapsed =
        std::chrono::duration<double>(
            currentTime - displayTimer).count();

    if (elapsed >= 1.0)
    {
        std::cout << "--------------------------------\n";

        std::cout << std::fixed << std::setprecision(2);

        std::cout << "Capture          : "
                  << get("capture")
                  << " ms\n";

        std::cout << "FaceMesh         : "
                  << get("face_mesh")
                  << " ms\n";

        std::cout << "EyeMonitor       : "
                  << get("eye_monitor")
                  << " ms\n";

        std::cout << "AttentionMonitor : "
                  << get("attention_monitor")
                  << " ms\n";

        std::cout << "Total            : "
                  << total()
                  << " ms\n";

        displayTimer = currentTime;
    }
}

}   // namespace dms