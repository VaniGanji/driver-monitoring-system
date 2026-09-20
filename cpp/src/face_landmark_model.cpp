#include "face_landmark_model.hpp"

#include <array>
#include <iostream>
#include <algorithm>

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
    // 1. Create square image without distorting the ROI
    //--------------------------------------------------------

    const int width =
        frame.cols;

    const int height =
        frame.rows;

    const int squareSize =
        std::max(width, height);

    const int padLeft =
        (squareSize - width) / 2;

    const int padRight =
        squareSize -
        width -
        padLeft;

    const int padTop =
        (squareSize - height) / 2;

    const int padBottom =
        squareSize -
        height -
        padTop;

    cv::Mat squareImage;

    cv::copyMakeBorder(
        frame,
        squareImage,
        padTop,
        padBottom,
        padLeft,
        padRight,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));

    //--------------------------------------------------------
    // 2. Resize square image to FAN2 input size
    //--------------------------------------------------------

    cv::Mat resized;

    cv::resize(
        squareImage,
        resized,
        cv::Size(
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE));

    //--------------------------------------------------------
    // 3. Convert image to float32
    //    Normalize pixel values to [0, 1]
    //--------------------------------------------------------

    cv::Mat floatImage;

    resized.convertTo(
        floatImage,
        CV_32FC3,
        1.0 / 255.0);

    //--------------------------------------------------------
    // 4. Convert OpenCV HWC → FAN2 NCHW
    //--------------------------------------------------------

    const int channelSize =
        MODEL_INPUT_SIZE *
        MODEL_INPUT_SIZE;

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
    // 5. Define FAN2 input tensor shape
    //
    //    1 × 3 × 256 × 256
    //--------------------------------------------------------

    std::array<int64_t, 4> inputShape =
    {
        1,
        MODEL_CHANNELS,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE
    };

    //--------------------------------------------------------
    // 6. Create CPU memory information
    //--------------------------------------------------------

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    //--------------------------------------------------------
    // 7. Create ONNX input tensor
    //--------------------------------------------------------

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size());

    //--------------------------------------------------------
    // 8. Get model input/output names
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
    // 9. Run FAN2 inference
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
    // 10. Get landmark output
    //--------------------------------------------------------

    float* landmarkData =
        outputs[0].GetTensorMutableData<float>();

    //--------------------------------------------------------
    // 11. FAN2 output coordinates
    //
    //     FAN2 coordinates are assumed to be in 64×64.
    //     64 → 256 means ×4.
    //
    //     The 256×256 image corresponds to squareImage.
    //--------------------------------------------------------

    const double squareScale =
        static_cast<double>(squareSize) /
        MODEL_INPUT_SIZE;

    //--------------------------------------------------------
    // 12. Map all landmarks back to original ROI
    //--------------------------------------------------------

    std::vector<cv::Point> landmarks;

    landmarks.reserve(
        LANDMARK_COUNT);

    for (int i = 0;
         i < LANDMARK_COUNT;
         ++i)
    {
        const float modelX =
            landmarkData[i * 3];

        const float modelY =
            landmarkData[i * 3 + 1];

        //----------------------------------------------------
        // FAN2 64×64 → 256×256
        //----------------------------------------------------

        const double squareX =
            modelX * 4.0;

        const double squareY =
            modelY * 4.0;

        //----------------------------------------------------
        // 256×256 → padded square image
        //----------------------------------------------------

        const double paddedX =
            squareX * squareScale;

        const double paddedY =
            squareY * squareScale;

        //----------------------------------------------------
        // Remove padding → original ROI coordinates
        //----------------------------------------------------

        const int originalX =
            static_cast<int>(
                paddedX -
                padLeft);

        const int originalY =
            static_cast<int>(
                paddedY -
                padTop);

        landmarks.emplace_back(
            originalX,
            originalY);
    }

    return landmarks;
}

}