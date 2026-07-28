#include <iostream>
#include "config.hpp"

int main()
{
    std::cout << "=================================\n";
    std::cout << "Driver Monitoring System\n";
    std::cout << "Modern C++ Reference Implementation\n";
    std::cout << "Version: 2.0\n";
    std::cout << "=================================\n";

    std::cout << "EAR Threshold : " << dms::EAR_THRESHOLD << std::endl;
    std::cout << "Closed Frames Threshold : " << dms::CLOSED_FRAMES_THRESHOLD << std::endl;

    return 0;
}