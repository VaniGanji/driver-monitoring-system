#include "eye_monitor.hpp"
#include "config.hpp"

#include <chrono>
#include <stdexcept>

namespace dms
{

EyeMonitor::EyeMonitor()
    : closedFrames_(0),
      blinkDetected_(false),
      blinkCount_(0),
      blinkCountResetTime_(std::chrono::steady_clock::now())
{
}


double EyeMonitor::calculateEAR(
    const std::vector<cv::Point>& landmarks,
    const std::array<int, 6>& eyeIndices)
{
    for (int index : eyeIndices)
    {
        if (index < 0 ||
            static_cast<std::size_t>(index) >= landmarks.size())
        {
            throw std::out_of_range("Invalid eye landmark index");
        }
    }

    const cv::Point& p0 = landmarks[eyeIndices[0]];
    const cv::Point& p1 = landmarks[eyeIndices[1]];
    const cv::Point& p2 = landmarks[eyeIndices[2]];
    const cv::Point& p3 = landmarks[eyeIndices[3]];
    const cv::Point& p4 = landmarks[eyeIndices[4]];
    const cv::Point& p5 = landmarks[eyeIndices[5]];

    const double v1 = cv::norm(p1 - p5);
    const double v2 = cv::norm(p2 - p4);
    const double h = cv::norm(p0 - p3);

    if (h <= 0.0)
    {
        return 0.0;
    }

    return (v1 + v2) / (2.0 * h);
}


EyeStateResult EyeMonitor::processEyeState(double averageEAR)
{
    // Eyes are considered closed.
    if (averageEAR < dms::EAR_THRESHOLD)
    {
        ++closedFrames_;
        blinkDetected_ = true;
    }
    else
    {
        // Eyes are open again.
        closedFrames_ = 0;

        // A closed period followed by an open frame = one blink.
        if (blinkDetected_)
        {
            ++blinkCount_;
            blinkDetected_ = false;
        }
    }

    // Reset blink count every 60 seconds.
    const auto now = std::chrono::steady_clock::now();

    const auto elapsedSeconds =
        std::chrono::duration_cast<std::chrono::seconds>(
            now - blinkCountResetTime_).count();

    if (elapsedSeconds >= dms::BLINK_COUNT_RESET_SECONDS)
    {
        blinkCount_ = 0;
        blinkCountResetTime_ = now;
    }

    const bool isDrowsy =
        closedFrames_ >= dms::CLOSED_FRAMES_THRESHOLD;

    return {blinkCount_, isDrowsy};
}

} // namespace dms