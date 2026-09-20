#include "face_detector.hpp"

#include <algorithm>
#include <stdexcept>

namespace dms
{

FaceDetector::FaceDetector(
    const std::string& modelPath,
    float confidenceThreshold,
    float nmsThreshold,
    int topK)
    : confidenceThreshold_(confidenceThreshold),
      nmsThreshold_(nmsThreshold),
      topK_(topK)
{
    detector_ = cv::FaceDetectorYN::create(
        modelPath,
        "",
        cv::Size(320, 320),
        confidenceThreshold_,
        nmsThreshold_,
        topK_);

    if (detector_.empty())
    {
        throw std::runtime_error(
            "Failed to create YuNet face detector");
    }
}

bool FaceDetector::detect(
    const cv::Mat& frame,
    FaceDetection& detection)
{
    if (frame.empty())
    {
        return false;
    }

    // YuNet must know the actual input image size.
    detector_->setInputSize(frame.size());

    cv::Mat faces;
    detector_->detect(frame, faces);

    if (faces.empty() || faces.rows == 0)
    {
        return false;
    }

    /*
     * YuNet output format:
     *
     * columns 0-3  : x, y, width, height
     * columns 4-13 : five facial landmarks
     * column 14    : confidence
     */

    int bestFaceIndex = -1;
    float bestConfidence = 0.0f;

    for (int i = 0; i < faces.rows; ++i)
    {
        const float confidence =
            faces.at<float>(i, 14);

        if (confidence > bestConfidence)
        {
            bestConfidence = confidence;
            bestFaceIndex = i;
        }
    }

    if (bestFaceIndex < 0)
    {
        return false;
    }

    const int x =
        static_cast<int>(faces.at<float>(bestFaceIndex, 0));

    const int y =
        static_cast<int>(faces.at<float>(bestFaceIndex, 1));

    const int width =
        static_cast<int>(faces.at<float>(bestFaceIndex, 2));

    const int height =
        static_cast<int>(faces.at<float>(bestFaceIndex, 3));

    cv::Rect faceBox(x, y, width, height);

    // Keep the bounding box inside the camera frame.
    faceBox &= cv::Rect(
        0,
        0,
        frame.cols,
        frame.rows);

    if (faceBox.width <= 0 || faceBox.height <= 0)
    {
        return false;
    }

    detection.boundingBox = faceBox;
    detection.confidence = bestConfidence;

    return true;
}

} // namespace dms