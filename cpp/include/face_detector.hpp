#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <string>

#include <opencv2/opencv.hpp>

namespace dms
{

struct FaceDetection
{
    cv::Rect boundingBox;
    float confidence;
};

class FaceDetector
{
public:
    FaceDetector(
        const std::string& modelPath,
        float confidenceThreshold = 0.6f,
        float nmsThreshold = 0.3f,
        int topK = 5000);

    bool detect(
        const cv::Mat& frame,
        FaceDetection& detection);

private:
    cv::Ptr<cv::FaceDetectorYN> detector_;
    float confidenceThreshold_;
    float nmsThreshold_;
    int topK_;
};

} // namespace dms

#endif