#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace dms
{

class PerformanceMonitor
{
public:

    PerformanceMonitor();

    void start(const std::string& stage);

    void stop(const std::string& stage);

    double get(const std::string& stage) const;

    double total() const;

    void reset();

    void displayPerformance();

private:

    using Clock = std::chrono::steady_clock;

    std::unordered_map<std::string, Clock::time_point> startTimes;

    std::unordered_map<std::string, double> elapsedTimes;

    Clock::time_point displayTimer;
};

}   // namespace dms