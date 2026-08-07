#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

#include <opencv2/opencv.hpp>

namespace dms
{

//------------------------------------------------------------
// Draw selected FaceMesh landmarks
//------------------------------------------------------------

void drawLandmarkDebug(
    cv::Mat& frame,
    const std::vector<cv::Point>& landmarks,
    bool showIds = true,
    bool showPoints = true);

//------------------------------------------------------------
// Alarm
//------------------------------------------------------------

void playAlarm();

//------------------------------------------------------------
// FPS
//------------------------------------------------------------

double calculateFPS(double& previousTime);

} // namespace dms

#endif