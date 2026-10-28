
#include "event_logger.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace dms
{

Logger::Logger()
    : logDirectory("results"),
      logFile("results/session_log_cpp.csv")
{
}

void Logger::initialize()
{

    try
    {
        std::filesystem::create_directories(logDirectory);

        const bool needsHeader =
            !std::filesystem::exists(logFile) ||
            std::filesystem::file_size(logFile) == 0;

        std::ofstream file(
            logFile,
            std::ios::out | std::ios::app);

        if (!file.is_open())
        {
            std::cerr
                << "ERROR: Could not open log file: "
                << logFile << '\n';
            initialized_ = false;
            return;
        }

        if (needsHeader)
        {
            file << "Timestamp,Event,Details\n";
        }

        file.flush();

        if (!file.good())
        {
            std::cerr << "ERROR: Could not initialize log file.\n";
            initialized_ = false;
            return;
        }

        initialized_ = true;

        std::cout
            << "Event logger initialized: "
            << logFile << '\n';
    }
    catch (const std::filesystem::filesystem_error& error)
    {
        std::cerr
            << "ERROR: Could not create log directory: "
            << error.what() << '\n';

        initialized_ = false;
    }
}

std::string Logger::currentTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};

#ifdef _WIN32
    localtime_s(&localTime, &time);
#else
    localtime_r(&time, &localTime);
#endif

    std::ostringstream timestamp;
    timestamp << std::put_time(
        &localTime, "%Y-%m-%d %H:%M:%S");

    return timestamp.str();
}

std::string Logger::escapeCsv(const std::string& value)
{
    std::string escaped = "\"";

    for (const char character : value)
    {
        if (character == '"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += character;
        }
    }

    escaped += '"';
    return escaped;
}

void Logger::logEvent(const std::string& event)
{
    if (!initialized_)
    {
        std::cerr
            << "WARNING: Event not logged because the logger "
            << "is not initialized: " << event << '\n';
        return;
    }

    std::ofstream file(
        logFile,
        std::ios::out | std::ios::app);

    if (!file.is_open())
    {
        std::cerr << "ERROR: Could not append to log file.\n";
        return;
    }

    file << currentTimestamp()
         << ',' << escapeCsv(event)
         << '\n';

    file.flush();

    if (!file.good())
    {
        std::cerr << "ERROR: Failed to write event to log file.\n";
    }
}

} // namespace dms
