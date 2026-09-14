#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

#include "config.hpp"
#include "event_logger.hpp"
#include "eye_monitor.hpp"
#include "face_landmark_model.hpp"

namespace
{
constexpr int LANDMARK_COUNT = 68;
constexpr int CAMERA_INDEX = 0;
constexpr int CAMERA_WIDTH = 640;
constexpr int CAMERA_HEIGHT = 480;

const std::string WINDOW_NAME = "Driver Monitoring System";

dms::EyeMonitor eyeMonitor;
}

int main()
{
    try
    {
        std::cout << "=================================\n";
        std::cout << "Driver Monitoring System\n";
        std::cout << "Modern C++ Reference Implementation\n";
        std::cout << "Version: 2.0\n";
        std::cout << "=================================\n";

        // Initialize logger
        dms::Logger logger;
        logger.initialize();
        logger.logEvent("Starting live camera DMS");

        // Initialize FAN2 landmark model once.
        // The ONNX Runtime session is reused for every camera frame.
        const std::string modelPath =
            "models/face_landmark/fan2_68_landmark.onnx";

        dms::FaceLandmarkModel faceLandmarkModel(modelPath);

        // Open camera
        cv::VideoCapture camera(CAMERA_INDEX);

        if (!camera.isOpened())
        {
            std::cerr << "ERROR: Could not open camera.\n";
            logger.logEvent("ERROR: Could not open camera");
            return 1;
        }

        // Request camera resolution
        camera.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);

        std::cout << "Camera opened successfully.\n";
        std::cout << "Press 'q' to quit.\n";

        logger.logEvent("Camera opened successfully");

        cv::Mat frame;

        while (true)
        {
            // Capture one frame
            camera >> frame;

            if (frame.empty())
            {
                std::cerr << "WARNING: Empty camera frame.\n";
                continue;
            }

            // Run FAN2 inference
            const std::vector<cv::Point> landmarks =
                faceLandmarkModel.infer(frame);

            if (landmarks.size() == LANDMARK_COUNT)
            {
                // Calculate EAR for both eyes
                const double leftEAR =
                    eyeMonitor.calculateEAR(
                        landmarks,
                        dms::LEFT_EYE);

                const double rightEAR =
                    eyeMonitor.calculateEAR(
                        landmarks,
                        dms::RIGHT_EYE);

                const double averageEAR =
                    (leftEAR + rightEAR) / 2.0;

                // Update temporal eye state
                const dms::EyeStateResult eyeState =
                    eyeMonitor.processEyeState(averageEAR);

                // Draw all 68 landmarks
                for (int i = 0; i < LANDMARK_COUNT; ++i)
                {
                    cv::circle(
                        frame,
                        landmarks[i],
                        2,
                        cv::Scalar(0, 255, 0),
                        -1);

                    // Display landmark ID
                    cv::putText(
                        frame,
                        std::to_string(i),
                        landmarks[i] + cv::Point(3, -3),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.35,
                        cv::Scalar(255, 255, 255),
                        1);
                }

                // Display EAR information
                cv::putText(
                    frame,
                    "Left EAR: " + std::to_string(leftEAR),
                    cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);

                cv::putText(
                    frame,
                    "Right EAR: " + std::to_string(rightEAR),
                    cv::Point(20, 60),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);

                cv::putText(
                    frame,
                    "Average EAR: " + std::to_string(averageEAR),
                    cv::Point(20, 90),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);

                // Display blink count
                cv::putText(
                    frame,
                    "Blink Count: " +
                        std::to_string(eyeState.blinkCount),
                    cv::Point(20, 125),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 255, 0),
                    2);

                // Display drowsiness state
                const std::string drowsyText =
                    eyeState.isDrowsy
                        ? "Drowsy: YES"
                        : "Drowsy: NO";

                const cv::Scalar drowsyColor =
                    eyeState.isDrowsy
                        ? cv::Scalar(0, 0, 255)
                        : cv::Scalar(0, 255, 0);

                cv::putText(
                    frame,
                    drowsyText,
                    cv::Point(20, 160),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    drowsyColor,
                    2);
            }
            else
            {
                // No valid 68-point landmark result
                cv::putText(
                    frame,
                    "No valid face landmarks",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 0, 255),
                    2);
            }

            // Display live frame
            cv::imshow(WINDOW_NAME, frame);

            // Exit when 'q' is pressed
            const int key = cv::waitKey(1);

            if (key == 'q' || key == 'Q')
            {
                break;
            }
        }

        // Release camera and close window
        camera.release();
        cv::destroyAllWindows();

        logger.logEvent("Live camera DMS stopped");

        std::cout << "\nLive camera test completed.\n";

        return 0;
    }
    catch (const Ort::Exception& exception)
    {
        std::cerr << "ONNX Runtime error: "
                  << exception.what()
                  << '\n';

        return 1;
    }
    catch (const cv::Exception& exception)
    {
        std::cerr << "OpenCV error: "
                  << exception.what()
                  << '\n';

        return 1;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: "
                  << exception.what()
                  << '\n';

        return 1;
    }
}