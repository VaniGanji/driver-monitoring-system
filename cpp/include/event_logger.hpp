#pragma once

#include <string>

namespace dms
{

class Logger
{
public:
    Logger();

    void initialize();

    void logEvent(const std::string& event);

private:
    std::string logDirectory;
    std::string logFile;

    bool initialized_ = false;

    static std::string currentTimestamp();
    static std::string escapeCsv(const std::string& value);
};

} // namespace dms