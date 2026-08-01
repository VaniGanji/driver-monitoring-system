#include <iostream>

#include <opencv2/opencv.hpp>

int main()
{
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));

    cv::putText(image,
                "OpenCV Smoke Test",
                cv::Point(120, 240),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                2);

    cv::circle(image,
               cv::Point(320, 150),
               40,
               cv::Scalar(255, 0, 0),
               3);

    cv::imshow("Smoke Test", image);

    std::cout << "Press any key in the OpenCV window..." << std::endl;

    cv::waitKey(0);

    return 0;
}