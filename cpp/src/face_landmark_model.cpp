#include "face_landmark_model.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace dms
{

FaceLandmarkModel::FaceLandmarkModel(
    const std::string& modelPath)
    : env_(
          ORT_LOGGING_LEVEL_WARNING,
          "DMS_FACE_LANDMARK"),
      sessionOptions_(),
      session_(
          nullptr)
{
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);

    session_ = Ort::Session(
        env_,
        modelPath.c_str(),
        sessionOptions_);
}

std::vector<cv::Point> FaceLandmarkModel::infer(
    const cv::Mat& frame)
{
    if (frame.empty())
    {
        std::cerr
            << "Error: Input frame is empty.\n";

        return {};
    }

    //--------------------------------------------------------
    // 1. Resize image to FAN2 input size
    //--------------------------------------------------------

    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE));

    //--------------------------------------------------------
    // 2. Convert image to float32
    //    Normalize pixel values to [0, 1]
    //--------------------------------------------------------

    cv::Mat floatImage;

    resized.convertTo(
        floatImage,
        CV_32FC3,
        1.0 / 255.0);

    //--------------------------------------------------------
    // 3. Convert OpenCV HWC → FAN2 NCHW
    //--------------------------------------------------------

    const int channelSize =
        MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

    std::vector<float> inputTensorValues(
        MODEL_CHANNELS * channelSize);

    for (int y = 0;
         y < MODEL_INPUT_SIZE;
         ++y)
    {
        for (int x = 0;
             x < MODEL_INPUT_SIZE;
             ++x)
        {
            const cv::Vec3f& pixel =
                floatImage.at<cv::Vec3f>(y, x);

            const int index =
                y * MODEL_INPUT_SIZE + x;

            inputTensorValues[index] =
                pixel[0];

            inputTensorValues[
                channelSize + index] =
                pixel[1];

            inputTensorValues[
                (2 * channelSize) + index] =
                pixel[2];
        }
    }

    //--------------------------------------------------------
    // 4. Define FAN2 input tensor shape
    //
    //     1 × 3 × 256 × 256
    //--------------------------------------------------------

    std::array<int64_t, 4> inputShape =
    {
        1,
        MODEL_CHANNELS,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE
    };

    //--------------------------------------------------------
    // 5. Create CPU memory information
    //--------------------------------------------------------

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    //--------------------------------------------------------
    // 6. Create ONNX input tensor
    //--------------------------------------------------------

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size());

    //--------------------------------------------------------
    // 7. Get model input/output names
    //--------------------------------------------------------

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName =
        session_.GetInputNameAllocated(
            0,
            allocator);

    auto outputName =
        session_.GetOutputNameAllocated(
            0,
            allocator);

    const char* inputNames[] =
    {
        inputName.get()
    };

    const char* outputNames[] =
    {
        outputName.get()
    };

    //--------------------------------------------------------
    // 8. Run FAN2 inference
    //--------------------------------------------------------

    auto outputs =
        session_.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);

    //--------------------------------------------------------
    // 9. Get landmark output data
    //--------------------------------------------------------

    float* landmarkData =
        outputs[0].GetTensorMutableData<float>();

    //--------------------------------------------------------
    // 10. Map FAN2 coordinates back to original image
    //--------------------------------------------------------

    const double scaleX =
        static_cast<double>(frame.cols) /
        MODEL_INPUT_SIZE;

    const double scaleY =
        static_cast<double>(frame.rows) /
        MODEL_INPUT_SIZE;

    std::vector<cv::Point> landmarks;

    landmarks.reserve(
        LANDMARK_COUNT);

    //--------------------------------------------------------
    // 11. Extract and map all 68 landmarks
    //--------------------------------------------------------

    for (int i = 0;
         i < LANDMARK_COUNT;
         ++i)
    {
        const float modelX =
            landmarkData[i * 3];

        const float modelY =
            landmarkData[i * 3 + 1];

        //----------------------------------------------------
        // FAN2 coordinates are in 64x64 space.
        //
        // 64 → 256 → original image
        //----------------------------------------------------

        const int originalX =
            static_cast<int>(
                modelX *
                4.0 *
                scaleX);

        const int originalY =
            static_cast<int>(
                modelY *
                4.0 *
                scaleY);

        landmarks.emplace_back(
            originalX,
            originalY);
    }

    return landmarks;
}

} // namespace dms