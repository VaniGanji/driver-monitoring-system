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

    // ============================================================
    // Detection Thresholds
    // ============================================================

    constexpr double EAR_THRESHOLD = 0.22;

    constexpr int CLOSED_FRAMES_THRESHOLD = 30;

    constexpr int BLINK_COUNT_RESET_SECONDS = 60;

    constexpr int HEAD_OFFSET_THRESHOLD = 30;

    constexpr int DISTRACTION_THRESHOLD = 60;

    // ============================================================
    // Debug Configuration
    // ============================================================

    // Landmark Visualization & Debugging
    constexpr bool SHOW_LANDMARK_IDS = false;
    constexpr bool SHOW_LANDMARK_POINTS = true;

    constexpr bool SHOW_PERFORMANCE = true;

} // namespace dms