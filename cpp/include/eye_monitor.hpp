#ifndef EYE_MONITOR_HPP
#define EYE_MONITOR_HPP

#include <array>
#include <chrono>
#include <vector>

#include <opencv2/opencv.hpp>

namespace dms
{

struct EyeStateResult
{
    int blinkCount;
    bool isDrowsy;
};

class EyeMonitor
{
public:
    EyeMonitor();

    double calculateEAR(
        const std::vector<cv::Point>& landmarks,
        const std::array<int, 6>& eyeIndices);

    EyeStateResult processEyeState(double averageEAR);

private:
    int closedFrames_;
    bool blinkDetected_;
    int blinkCount_;

    std::chrono::steady_clock::time_point blinkCountResetTime_;
};

} // namespace dms

#endif