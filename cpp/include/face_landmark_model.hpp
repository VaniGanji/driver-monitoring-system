#ifndef FACE_LANDMARK_MODEL_HPP
#define FACE_LANDMARK_MODEL_HPP

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace dms
{

class FaceLandmarkModel
{
public:
    explicit FaceLandmarkModel(const std::string& modelPath);

    std::vector<cv::Point> infer(const cv::Mat& frame);

private:
    static constexpr int MODEL_INPUT_SIZE = 256;
    static constexpr int MODEL_CHANNELS = 3;
    static constexpr int LANDMARK_COUNT = 68;

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::Session session_;
};

} // namespace dms

#endif