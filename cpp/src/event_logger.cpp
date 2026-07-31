#include "event_logger.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdlib>

namespace dms
{

Logger::Logger()
    : logDirectory("../../results"),
      logFile("../../results/session_log_cpp.csv")
{
}

void Logger::initialize()
{
    std::system(("mkdir -p " + logDirectory).c_str());

    std::ifstream checkFile(logFile);

    if (!checkFile.good())
    {
        std::ofstream file(logFile);

        file << "Timestamp,Event\n";
    }
}

void Logger::logEvent(const std::string& event)
{
    std::ofstream file(logFile, std::ios::app);

    auto now = std::chrono::system_clock::now();

    std::time_t currentTime =
        std::chrono::system_clock::to_time_t(now);

    file << std::put_time(std::localtime(&currentTime),
                          "%Y-%m-%d %H:%M:%S")
         << ","
         << event
         << '\n';
}

} // namespace dms