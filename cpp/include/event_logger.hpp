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
};

} // namespace dms