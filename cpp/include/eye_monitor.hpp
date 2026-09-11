#ifndef EYE_MONITOR_HPP
#define EYE_MONITOR_HPP

#include <array>
#include <vector>
#include <opencv2/opencv.hpp>

namespace dms
{

double calculateEAR(
    const std::vector<cv::Point>& landmarks,
    const std::array<int, 6>& eyeIndices);

} // namespace dms

#endif