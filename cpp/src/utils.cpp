#include "utils.hpp"

#include <cstdlib>

namespace dms
{


void drawLandmarkDebug(
    cv::Mat& frame,
    const std::vector<cv::Point>& landmarks,
    bool showIds,
    bool showPoints)
{
    for (std::size_t idx = 0; idx < landmarks.size(); ++idx)
    {
        const cv::Point& point = landmarks[idx];

        if (showPoints)
        {
            cv::circle(
                frame,
                point,
                2,
                cv::Scalar(0, 255, 0),
                cv::FILLED
            );
        }

        if (showIds)
        {
            cv::putText(
                frame,
                std::to_string(idx),
                cv::Point(point.x + 3, point.y - 3),
                cv::FONT_HERSHEY_SIMPLEX,
                0.35,
                cv::Scalar(0, 255, 255),
                1,
                cv::LINE_AA
            );
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