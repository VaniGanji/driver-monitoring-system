#include "eye_monitor.hpp"

#include <cmath>
#include <stdexcept>

namespace dms
{

double calculateEAR(
    const std::vector<cv::Point>& landmarks,
    const std::array<int, 6>& eyeIndices)
{
    // Make sure all six landmark indices are available.
    for (int index : eyeIndices)
    {
        if (index < 0 ||
            static_cast<std::size_t>(index) >= landmarks.size())
        {
            throw std::out_of_range("Invalid eye landmark index");
        }
    }

    const cv::Point2f& p0 = landmarks[eyeIndices[0]];
    const cv::Point2f& p1 = landmarks[eyeIndices[1]];
    const cv::Point2f& p2 = landmarks[eyeIndices[2]];
    const cv::Point2f& p3 = landmarks[eyeIndices[3]];
    const cv::Point2f& p4 = landmarks[eyeIndices[4]];
    const cv::Point2f& p5 = landmarks[eyeIndices[5]];

    // Vertical eye distances
    const double v1 = cv::norm(p1 - p5);
    const double v2 = cv::norm(p2 - p4);

    // Horizontal eye distance
    const double h = cv::norm(p0 - p3);

    // Avoid division by zero.
    if (h <= 0.0)
    {
        return 0.0;
    }

    // Eye Aspect Ratio
    return (v1 + v2) / (2.0 * h);
}

} // namespace dms