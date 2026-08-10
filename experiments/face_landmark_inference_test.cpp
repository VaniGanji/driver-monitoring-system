#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

int main()
{
    const std::string model_path =
        "models/face_landmark/fan2_68_landmark.onnx";

    const std::string image_path =
        "experiments/data/test_face.jpeg";

    const std::string output_path =
        "experiments/data/face_landmark_result.jpeg";

    try
    {
        // ------------------------------------------------------------
        // 1. Load image
        // ------------------------------------------------------------

        cv::Mat image = cv::imread(image_path);

        if (image.empty())
        {
            std::cerr << "Failed to load image: "
                      << image_path << '\n';
            return 1;
        }

        std::cout << "Image loaded: "
                  << image.cols << " x "
                  << image.rows << '\n';

        // ------------------------------------------------------------
        // 2. Resize image to model input size
        // ------------------------------------------------------------

        cv::Mat resized;

        cv::resize(
            image,
            resized,
            cv::Size(256, 256)
        );

        // ------------------------------------------------------------
        // 3. Convert to float32 and normalize
        // ------------------------------------------------------------

        cv::Mat float_image;

        resized.convertTo(
            float_image,
            CV_32FC3,
            1.0 / 255.0
        );

        // ------------------------------------------------------------
        // 4. HWC -> CHW
        // ------------------------------------------------------------

        constexpr int IMAGE_SIZE = 256;
        constexpr int CHANNELS = 3;

        const int channel_size =
            IMAGE_SIZE * IMAGE_SIZE;

        std::vector<float> input_tensor_values(
            CHANNELS * channel_size
        );

        for (int y = 0; y < IMAGE_SIZE; ++y)
        {
            for (int x = 0; x < IMAGE_SIZE; ++x)
            {
                const cv::Vec3f& pixel =
                    float_image.at<cv::Vec3f>(y, x);

                const int index =
                    y * IMAGE_SIZE + x;

                input_tensor_values[index] =
                    pixel[0];

                input_tensor_values[
                    channel_size + index] =
                    pixel[1];

                input_tensor_values[
                    2 * channel_size + index] =
                    pixel[2];
            }
        }

        // ------------------------------------------------------------
        // 5. Create ONNX Runtime environment
        // ------------------------------------------------------------

        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "DMS_FACE_LANDMARK_TEST"
        );

        Ort::SessionOptions session_options;

        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_BASIC
        );

        // ------------------------------------------------------------
        // 6. Load model
        // ------------------------------------------------------------

        Ort::Session session(
            env,
            model_path.c_str(),
            session_options
        );

        Ort::AllocatorWithDefaultOptions allocator;

        // ------------------------------------------------------------
        // 7. Create input tensor
        // ------------------------------------------------------------

        std::array<int64_t, 4> input_shape =
        {
            1, 3, 256, 256
        };

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_shape.data(),
                input_shape.size()
            );

        // ------------------------------------------------------------
        // 8. Get input/output names
        // ------------------------------------------------------------

        auto input_name =
            session.GetInputNameAllocated(
                0,
                allocator
            );

        auto output_name =
            session.GetOutputNameAllocated(
                0,
                allocator
            );

        const char* input_names[] =
        {
            input_name.get()
        };

        const char* output_names[] =
        {
            output_name.get()
        };

        // ------------------------------------------------------------
        // 9. Run inference
        // ------------------------------------------------------------

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // ------------------------------------------------------------
        // 10. Read landmarks
        // ------------------------------------------------------------

        float* landmark_data =
            outputs[0].GetTensorMutableData<float>();

        // ------------------------------------------------------------
        // 11. Draw landmarks
        //
        // FAN2 landmark coordinates are in 64x64 space.
        // 256 / 64 = 4.
        // ------------------------------------------------------------

        constexpr float LANDMARK_SCALE = 4.0f;

        cv::Mat result = resized.clone();

        for (int i = 0; i < 68; ++i)
        {
            const float model_x =
                landmark_data[i * 3 + 0];

            const float model_y =
                landmark_data[i * 3 + 1];

            const float score =
                landmark_data[i * 3 + 2];

            const int x = static_cast<int>(
                model_x * LANDMARK_SCALE
            );

            const int y = static_cast<int>(
                model_y * LANDMARK_SCALE
            );

            // Only draw points that fall inside
            // the 256x256 image.
            if (x >= 0 && x < result.cols &&
                y >= 0 && y < result.rows)
            {
                cv::circle(
                    result,
                    cv::Point(x, y),
                    2,
                    cv::Scalar(0, 255, 0),
                    -1
                );

                std::cout
                    << "Landmark "
                    << i
                    << ": x=" << x
                    << ", y=" << y
                    << ", score=" << score
                    << '\n';
            }
        }

        // ------------------------------------------------------------
        // 12. Save result
        // ------------------------------------------------------------

        if (!cv::imwrite(output_path, result))
        {
            std::cerr << "Failed to save result: "
                      << output_path << '\n';

            return 1;
        }

        std::cout << "\nLandmark visualization saved to:\n"
                  << output_path
                  << '\n';
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "\nONNX Runtime error:\n"
                  << e.what()
                  << '\n';

        return 1;
    }

    return 0;
}