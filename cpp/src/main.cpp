#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "event_logger.hpp"
#include "eye_monitor.hpp"
#include "face_landmark_model.hpp"

dms::EyeMonitor eyeMonitor;

namespace
{

//------------------------------------------------------------
// Application configuration
//------------------------------------------------------------

const std::string MODEL_PATH =
    "models/face_landmark/fan2_68_landmark.onnx";

const std::string TEST_IMAGE_PATH =
    "experiments/data/test_face.jpeg";

const std::string OUTPUT_IMAGE_PATH =
    "experiments/data/result_test_face.jpeg";

constexpr int LANDMARK_COUNT = 68;

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
            "Application Started");


        //----------------------------------------------------
        // 2. Initialize FAN2 face landmark model
        //
        // The model is loaded once and reused.
        //----------------------------------------------------

        dms::FaceLandmarkModel faceLandmarkModel(
            MODEL_PATH);


        //----------------------------------------------------
        // 3. Load test image
        //
        // This is temporary until camera integration.
        //----------------------------------------------------

        cv::Mat frame =
            cv::imread(
                TEST_IMAGE_PATH);


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
        // 4. Run FAN2 inference
        //----------------------------------------------------

        const std::vector<cv::Point> landmarks =
            faceLandmarkModel.infer(frame);


        //----------------------------------------------------
        // 5. Validate landmark count
        //----------------------------------------------------

        if (landmarks.size() != LANDMARK_COUNT)
        {
            std::cerr
                << "Face landmark inference failed.\n";

            return 1;
        }


        logger.logEvent(
            "68 Face Landmarks Detected");


        //----------------------------------------------------
        // 6. Calculate EAR
        //----------------------------------------------------

        double leftEAR =
            eyeMonitor.calculateEAR(
                landmarks,
                dms::LEFT_EYE);


        double rightEAR =
            eyeMonitor.calculateEAR(
                landmarks,
                dms::RIGHT_EYE);


        double averageEAR =
            (leftEAR + rightEAR) / 2.0;


        //----------------------------------------------------
        // 7. Process eye state
        //----------------------------------------------------

        dms::EyeStateResult eyeState =
            eyeMonitor.processEyeState(
                averageEAR);


        //----------------------------------------------------
        // 8. Print eye state
        //----------------------------------------------------

        std::cout
            << "Left EAR    : "
            << leftEAR
            << '\n';

        std::cout
            << "Right EAR   : "
            << rightEAR
            << '\n';

        std::cout
            << "Average EAR : "
            << averageEAR
            << '\n';

        std::cout
            << "Blink Count : "
            << eyeState.blinkCount
            << '\n';

        std::cout
            << "Drowsy      : "
            << (eyeState.isDrowsy ? "YES" : "NO")
            << '\n';


        //----------------------------------------------------
        // 9. Create visualization image
        //
        // IMPORTANT:
        //
        // 'frame' is the ORIGINAL image.
        //
        // FAN2 landmarks returned by infer()
        // are already mapped to the original
        // image coordinate system.
        //----------------------------------------------------

        cv::Mat result =
            frame.clone();


        //----------------------------------------------------
        // 10. Draw landmarks
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
                    0),
                -1);


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
                    0),
                1);
        }


        //----------------------------------------------------
        // 11. Draw EAR
        //----------------------------------------------------

        std::string earText =
            cv::format(
                "EAR: %.3f",
                averageEAR);


        cv::putText(
            result,
            earText,
            cv::Point(
                30,
                40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(
                0,
                255,
                0),
            2);


        //----------------------------------------------------
        // 12. Save visualization
        //----------------------------------------------------

        if (!cv::imwrite(
                OUTPUT_IMAGE_PATH,
                result))
        {
            std::cerr
                << "Failed to save result image.\n";

            return 1;
        }


        //----------------------------------------------------
        // 13. Print result
        //----------------------------------------------------

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


    //--------------------------------------------------------
    // Exception handling
    //--------------------------------------------------------

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