
#include "attention_monitor.hpp"

#include <cmath>
#include <cstddef>

namespace dms
{

std::string estimateHeadDirection(
    const std::vector<cv::Point>& landmarks)
{
    constexpr std::size_t NOSE_TIP = 30;
    constexpr std::size_t LEFT_EYE_OUTER = 36;
    constexpr std::size_t RIGHT_EYE_OUTER = 45;

    // Fixed values based on your camera observations.
    constexpr float NEUTRAL_OFFSET = 0.14f;

    // Separate thresholds allow independent tuning.
    constexpr float LEFT_THRESHOLD = 0.10f;
    constexpr float RIGHT_THRESHOLD = 0.35f;

    if (landmarks.size() <= RIGHT_EYE_OUTER)
    {
        return "Unknown";
    }

    const float noseX =
        static_cast<float>(landmarks[NOSE_TIP].x);

    const float leftEyeX =
        static_cast<float>(landmarks[LEFT_EYE_OUTER].x);

    const float rightEyeX =
        static_cast<float>(landmarks[RIGHT_EYE_OUTER].x);

    const float eyeCenterX =
        (leftEyeX + rightEyeX) / 2.0f;

    const float eyeWidth =
        std::abs(rightEyeX - leftEyeX);

    if (eyeWidth < 1.0f)
    {
        return "Unknown";
    }

    // Normalize using current eye width.
    const float normalizedOffset =
        (noseX - eyeCenterX) / eyeWidth;

    // Compensate for the observed forward-facing bias.
    const float centeredOffset =
        normalizedOffset - NEUTRAL_OFFSET;

    if (centeredOffset > LEFT_THRESHOLD)
    {
        return "Head Left";
    }

    if (centeredOffset < -RIGHT_THRESHOLD)
    {
        return "Head Right";
    }

    return "Head Forward";
}

} // namespace dms
