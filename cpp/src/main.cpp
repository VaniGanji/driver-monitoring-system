#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

#include "config.hpp"
#include "event_logger.hpp"
#include "eye_monitor.hpp"
#include "face_detector.hpp"
#include "face_landmark_model.hpp"

namespace
{

constexpr int CAMERA_INDEX = 0;
constexpr int CAMERA_WIDTH = 640;
constexpr int CAMERA_HEIGHT = 480;

const std::string WINDOW_NAME =
    "Driver Monitoring System";

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

        const std::string detectorModelPath =
            "models/face_detection/"
            "face_detection_yunet_2026may.onnx";

        const std::string landmarkModelPath =
            "models/face_landmark/"
            "fan2_68_landmark.onnx";

        // Initialize face detector once.
        dms::FaceDetector faceDetector(
            detectorModelPath,
            0.6f,
            0.3f,
            5000);

        // Initialize FAN2 once.
        dms::FaceLandmarkModel faceLandmarkModel(
            landmarkModelPath);

        // Open camera.
        cv::VideoCapture camera(CAMERA_INDEX);

        if (!camera.isOpened())
        {
            std::cerr << "ERROR: Could not open camera.\n";
            return 1;
        }

        camera.set(
            cv::CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH);

        camera.set(
            cv::CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT);

        std::cout << "Camera opened successfully.\n";
        std::cout << "Press 'q' to quit.\n";

        cv::Mat frame;

        while (true)
        {
            camera >> frame;

            if (frame.empty())
            {
                continue;
            }

            // --------------------------------------------------
            // 1. Detect face
            // --------------------------------------------------

            dms::FaceDetection faceDetection;

            const bool faceDetected =
                faceDetector.detect(
                    frame,
                    faceDetection);

            if (faceDetected)
            {
                const cv::Rect faceBox =
                    faceDetection.boundingBox;

                //--------------------------------------------------------
                // Validate YuNet bounding box
                //--------------------------------------------------------

                if (faceBox.width <= 0 ||
                    faceBox.height <= 0 ||
                    faceBox.x < 0 ||
                    faceBox.y < 0 ||
                    faceBox.x + faceBox.width > frame.cols ||
                    faceBox.y + faceBox.height > frame.rows)
                {
                    continue;
                }

                //--------------------------------------------------------
                // Extract face ROI
                //--------------------------------------------------------

                const cv::Mat faceROI =
                    frame(faceBox);

                //--------------------------------------------------------
                // FAN2 landmark inference
                //
                // Returned landmarks are ROI-relative.
                //--------------------------------------------------------

                const std::vector<cv::Point> landmarks =
                    faceLandmarkModel.infer(faceROI);

                //--------------------------------------------------------
                // Debug output
                //--------------------------------------------------------

                if (!landmarks.empty())
                {
                    const int debugIndices[] =
                    {
                        0, 10, 20, 30, 40, 50, 60, 67
                    };

                    std::cout << "LIVE landmarks: ";

                    for (int index : debugIndices)
                    {
                        std::cout
                            << "[" << index << ": "
                            << landmarks[index].x
                            << ","
                            << landmarks[index].y
                            << "] ";
                    }

                    std::cout << "\n";
                }

                //--------------------------------------------------------
                // Draw YuNet bounding box
                //--------------------------------------------------------

                cv::rectangle(
                    frame,
                    faceBox,
                    cv::Scalar(255, 0, 0),
                    2);

                //--------------------------------------------------------
                // Convert ROI coordinates → frame coordinates
                //--------------------------------------------------------

                for (const cv::Point& point : landmarks)
                {
                    const cv::Point framePoint =
                        point +
                        cv::Point(
                            faceBox.x,
                            faceBox.y);

                    cv::circle(
                        frame,
                        framePoint,
                        2,
                        cv::Scalar(0, 255, 0),
                        -1);
                }

                // --------------------------------------------------
                // 6. Calculate EAR
                // --------------------------------------------------

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

                // --------------------------------------------------
                // 7. Process temporal eye state
                // --------------------------------------------------

                const dms::EyeStateResult eyeState =
                    eyeMonitor.processEyeState(
                        averageEAR);

                // --------------------------------------------------
                // 9. Display EAR
                // --------------------------------------------------

                cv::putText(
                    frame,
                    "Left EAR: " +
                        std::to_string(leftEAR),
                    cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);

                cv::putText(
                    frame,
                    "Right EAR: " +
                        std::to_string(rightEAR),
                    cv::Point(20, 60),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);

                cv::putText(
                    frame,
                    "Average EAR: " +
                        std::to_string(averageEAR),
                    cv::Point(20, 90),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);

                // --------------------------------------------------
                // 10. Blink count
                // --------------------------------------------------

                cv::putText(
                    frame,
                    "Blink Count: " +
                        std::to_string(
                            eyeState.blinkCount),
                    cv::Point(20, 125),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 255, 0),
                    2);

                // --------------------------------------------------
                // 11. Drowsiness
                // --------------------------------------------------

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
                cv::putText(
                    frame,
                    "No face detected",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 0, 255),
                    2);
            }

            // --------------------------------------------------
            // Display
            // --------------------------------------------------

            cv::imshow(
                WINDOW_NAME,
                frame);

            const int key =
                cv::waitKey(1);

            if (key == 'q' || key == 'Q')
            {
                break;
            }
        }

        camera.release();
        cv::destroyAllWindows();

        std::cout << "\nLive camera test completed.\n";

        return 0;
    }
    catch (const Ort::Exception& exception)
    {
        std::cerr
            << "ONNX Runtime error: "
            << exception.what()
            << '\n';

        return 1;
    }
    catch (const cv::Exception& exception)
    {
        std::cerr
            << "OpenCV error: "
            << exception.what()
            << '\n';

        return 1;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "Error: "
            << exception.what()
            << '\n';

        return 1;
    }
}