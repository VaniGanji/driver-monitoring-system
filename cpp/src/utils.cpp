#include "utils.hpp"

#include <cstdlib>

namespace dms
{

namespace
{

const std::vector<int> IMPORTANT_POINTS =
{
    1,
    33,133,
    362,263,
    469,470,471,472,
    474,475,476,477
};

} // anonymous namespace

void drawLandmarkDebug(
    cv::Mat& frame,
    const std::vector<cv::Point>& landmarks,
    bool showIds,
    bool showPoints)
{
    for (int idx : IMPORTANT_POINTS)
    {
        if (idx >= landmarks.size())
            continue;

        const cv::Point& point = landmarks[idx];

        if (showPoints)
        {
            cv::circle(frame,
                       point,
                       4,
                       cv::Scalar(0, 0, 255),
                       -1);
        }

        if (showIds)
        {
            cv::putText(frame,
                        std::to_string(idx),
                        point,
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.3,
                        cv::Scalar(0, 255, 255),
                        1);
        }
    }
}

void playAlarm()
{
#ifdef __APPLE__
    std::system("afplay assets/alarm.wav");
#else
    std::system("aplay assets/alarm.wav");
#endif
}

double calculateFPS(double& previousTime)
{
    const double currentTime =
        static_cast<double>(cv::getTickCount()) /
        cv::getTickFrequency();

    const double timeDiff = currentTime - previousTime;

    previousTime = currentTime;

    if (timeDiff > 0.0)
    {
        return 1.0 / timeDiff;
    }

    return 0.0;
}

} // namespace dms