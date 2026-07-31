#include <iostream>

#include "event_logger.hpp"

int main()
{
    std::cout << "=================================\n";
    std::cout << "Driver Monitoring System\n";
    std::cout << "Modern C++ Reference Implementation\n";
    std::cout << "Version: 2.0\n";
    std::cout << "=================================\n";

    dms::Logger logger;

    logger.initialize();

    logger.logEvent("Application Started");
    logger.logEvent("Camera Initialized");
    logger.logEvent("Face Detected");

    std::cout << "Logger test completed." << std::endl;

    return 0;
}