#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

#include "event_logger.hpp"

namespace
{

//------------------------------------------------------------
// FAN2 model configuration
//------------------------------------------------------------

constexpr int MODEL_INPUT_SIZE = 256;
constexpr int MODEL_CHANNELS = 3;
constexpr int LANDMARK_COUNT = 68;

const std::string MODEL_PATH =
    "models/face_landmark/fan2_68_landmark.onnx";

const std::string TEST_IMAGE_PATH =
    "experiments/data/test_face.jpeg";

const std::string OUTPUT_IMAGE_PATH =
    "experiments/data/result_test_face.jpeg";


//------------------------------------------------------------
// FAN2 face landmark inference
//------------------------------------------------------------

std::vector<cv::Point> runFaceLandmarkInference(
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
    //
    // Original image:
    //     e.g. 768 x 1024
    //
    // FAN2 input:
    //     256 x 256
    //--------------------------------------------------------

    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE
        )
    );


    //--------------------------------------------------------
    // 2. Convert image to float32
    //    Normalize pixel values to [0, 1]
    //--------------------------------------------------------

    cv::Mat floatImage;

    resized.convertTo(
        floatImage,
        CV_32FC3,
        1.0 / 255.0
    );


    //--------------------------------------------------------
    // 3. Convert OpenCV HWC → FAN2 NCHW
    //
    // OpenCV:
    //     Height × Width × Channels
    //
    // FAN2:
    //     1 × Channels × Height × Width
    //--------------------------------------------------------

    const int channelSize =
        MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

    std::vector<float> inputTensorValues(
        MODEL_CHANNELS * channelSize
    );

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

            // Channel 0
            inputTensorValues[index] =
                pixel[0];

            // Channel 1
            inputTensorValues[
                channelSize + index] =
                pixel[1];

            // Channel 2
            inputTensorValues[
                (2 * channelSize) + index] =
                pixel[2];
        }
    }


    //--------------------------------------------------------
    // 4. Create ONNX Runtime environment
    //--------------------------------------------------------

    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "DMS_FACE_LANDMARK"
    );


    //--------------------------------------------------------
    // 5. Configure ONNX Runtime session
    //--------------------------------------------------------

    Ort::SessionOptions sessionOptions;

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC
    );


    //--------------------------------------------------------
    // 6. Load FAN2 model
    //--------------------------------------------------------

    Ort::Session session(
        env,
        MODEL_PATH.c_str(),
        sessionOptions
    );


    //--------------------------------------------------------
    // 7. Define FAN2 input tensor shape
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
    // 8. Create CPU memory information
    //--------------------------------------------------------

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );


    //--------------------------------------------------------
    // 9. Create ONNX input tensor
    //--------------------------------------------------------

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size()
        );


    //--------------------------------------------------------
    // 10. Get model input/output names
    //--------------------------------------------------------

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName =
        session.GetInputNameAllocated(
            0,
            allocator
        );

    auto outputName =
        session.GetOutputNameAllocated(
            0,
            allocator
        );

    const char* inputNames[] =
    {
        inputName.get()
    };

    const char* outputNames[] =
    {
        outputName.get()
    };


    //--------------------------------------------------------
    // 11. Run FAN2 inference
    //
    // Output 0:
    //
    // landmarks_xyscore
    //
    // Shape:
    //     1 × 68 × 3
    //
    // Each landmark:
    //     X
    //     Y
    //     Score
    //--------------------------------------------------------

    auto outputs =
        session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );


    //--------------------------------------------------------
    // 12. Get landmark output data
    //--------------------------------------------------------

    float* landmarkData =
        outputs[0].GetTensorMutableData<float>();


    //--------------------------------------------------------
    // 13. Map FAN2 coordinates back to original image
    //
    // FAN2 landmark coordinates:
    //     64 × 64 heatmap coordinate system
    //
    // First:
    //     64 → 256
    //
    // Then:
    //     256 → original image
    //--------------------------------------------------------

    const double scaleX =
        static_cast<double>(frame.cols) /
        MODEL_INPUT_SIZE;

    const double scaleY =
        static_cast<double>(frame.rows) /
        MODEL_INPUT_SIZE;


    std::vector<cv::Point> landmarks;

    landmarks.reserve(
        LANDMARK_COUNT
    );


    //--------------------------------------------------------
    // 14. Extract and map all 68 landmarks
    //--------------------------------------------------------

    for (int i = 0;
         i < LANDMARK_COUNT;
         ++i)
    {
        const float modelX =
            landmarkData[i * 3];

        const float modelY =
            landmarkData[i * 3 + 1];

        const float score =
            landmarkData[i * 3 + 2];


        //----------------------------------------------------
        // FAN2 coordinates are in 64x64 space.
        //
        // Multiply by 4:
        //
        //     64 → 256
        //
        // Then scale to original image.
        //----------------------------------------------------

        const int originalX =
            static_cast<int>(
                modelX *
                4.0 *
                scaleX
            );

        const int originalY =
            static_cast<int>(
                modelY *
                4.0 *
                scaleY
            );


        landmarks.emplace_back(
            originalX,
            originalY
        );


        //----------------------------------------------------
        // Print landmark information
        //----------------------------------------------------

        std::cout
            << "Landmark "
            << i
            << ": x="
            << originalX
            << ", y="
            << originalY
            << ", score="
            << score
            << '\n';
    }


    return landmarks;
}

} // anonymous namespace


//------------------------------------------------------------
// Main
//------------------------------------------------------------

int main()
{
    std::cout
        << "=================================\n";

    std::cout
        << "Driver Monitoring System\n";

    std::cout
        << "Modern C++ Reference Implementation\n";

    std::cout
        << "Version: 2.0\n";

    std::cout
        << "=================================\n";


    try
    {
        //----------------------------------------------------
        // 1. Initialize logger
        //----------------------------------------------------

        dms::Logger logger;

        logger.initialize();

        logger.logEvent(
            "Application Started"
        );


        //----------------------------------------------------
        // 2. Load test image
        //
        // This is temporary until camera integration.
        //----------------------------------------------------

        cv::Mat frame =
            cv::imread(
                TEST_IMAGE_PATH
            );


        if (frame.empty())
        {
            std::cerr
                << "Failed to load test image: "
                << TEST_IMAGE_PATH
                << '\n';

            return 1;
        }


        std::cout
            << "Test image loaded: "
            << frame.cols
            << " x "
            << frame.rows
            << '\n';


        //----------------------------------------------------
        // 3. Run FAN2 inference
        //----------------------------------------------------

        const std::vector<cv::Point> landmarks =
            runFaceLandmarkInference(
                frame
            );


        //----------------------------------------------------
        // 4. Validate landmark count
        //----------------------------------------------------

        if (landmarks.size() != LANDMARK_COUNT)
        {
            std::cerr
                << "Face landmark inference failed.\n";

            return 1;
        }


        logger.logEvent(
            "68 Face Landmarks Detected"
        );


        //----------------------------------------------------
        // 5. Create visualization image
        //
        // IMPORTANT:
        //
        // 'frame' is the ORIGINAL image.
        //
        // The landmarks returned by
        // runFaceLandmarkInference()
        // are already mapped to the original
        // image coordinate system.
        //----------------------------------------------------

        cv::Mat result =
            frame.clone();


        //----------------------------------------------------
        // 6. Draw landmarks
        //----------------------------------------------------

        for (int i = 0;
             i < static_cast<int>(
                     landmarks.size());
             ++i)
        {
            //------------------------------------------------
            // Draw landmark point
            //------------------------------------------------

            cv::circle(
                result,
                landmarks[i],
                4,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                -1
            );


            //------------------------------------------------
            // Draw landmark ID
            //------------------------------------------------

            cv::putText(
                result,
                std::to_string(i),
                landmarks[i] +
                    cv::Point(5, -5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                1
            );
        }


        //----------------------------------------------------
        // 7. Save visualization
        //----------------------------------------------------

        if (!cv::imwrite(
                OUTPUT_IMAGE_PATH,
                result))
        {
            std::cerr
                << "Failed to save result image.\n";

            return 1;
        }


        std::cout
            << "\nFAN2 integration successful.\n";

        std::cout
            << "Detected "
            << landmarks.size()
            << " face landmarks.\n";

        std::cout
            << "Result image saved to: "
            << OUTPUT_IMAGE_PATH
            << '\n';


        std::cout
            << "\nLogger test completed.\n";
    }
    catch (const Ort::Exception& e)
    {
        std::cerr
            << "\nONNX Runtime error:\n"
            << e.what()
            << '\n';

        return 1;
    }
    catch (const cv::Exception& e)
    {
        std::cerr
            << "\nOpenCV error:\n"
            << e.what()
            << '\n';

        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "\nApplication error:\n"
            << e.what()
            << '\n';

        return 1;
    }


    return 0;
}