#pragma once

#include <array>

namespace dms
{
    // ============================================================
    // MediaPipe Face Mesh Landmark Indices
    // ============================================================

    constexpr std::array<int, 6> LEFT_EYE{
        36, 37, 38, 39, 40, 41
    };

    constexpr std::array<int, 6> RIGHT_EYE{
        42, 43, 44, 45, 46, 47
    };

    constexpr std::array<int, 4> LEFT_IRIS{
        474, 475, 476, 477
    };

    constexpr std::array<int, 4> RIGHT_IRIS{
        469, 470, 471, 472
    };

    constexpr int LEFT_EYE_LEFT  = 33;
    constexpr int LEFT_EYE_RIGHT = 133;

    constexpr int RIGHT_EYE_LEFT  = 362;
    constexpr int RIGHT_EYE_RIGHT = 263;

    constexpr int NOSE_TIP = 1;

    // ============================================================
    // Detection Thresholds
    // ============================================================

    constexpr double EAR_THRESHOLD = 0.22;

    constexpr int CLOSED_FRAMES_THRESHOLD = 30;

    constexpr int HEAD_OFFSET_THRESHOLD = 30;

    constexpr int DISTRACTION_THRESHOLD = 60;

    constexpr double GAZE_LEFT_THRESHOLD  = 0.30;
    constexpr double GAZE_RIGHT_THRESHOLD = 0.70;

    // ============================================================
    // Debug Configuration
    // ============================================================

    constexpr bool SHOW_LANDMARK_IDS = false;

    constexpr bool SHOW_PERFORMANCE = true;

} // namespace dms