#include <iostream>
#include <string>

#include <onnxruntime/onnxruntime_cxx_api.h>

int main()
{
    const std::string model_path =
        "models/face_landmark/fan2_68_landmark.onnx";

    try
    {
        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "DMS_MODEL_INSPECT"
        );

        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_BASIC
        );

        Ort::Session session(
            env,
            model_path.c_str(),
            session_options
        );

        Ort::AllocatorWithDefaultOptions allocator;

        const size_t input_count = session.GetInputCount();
        const size_t output_count = session.GetOutputCount();

        std::cout << "=================================\n";
        std::cout << "ONNX Face Landmark Model Inspection\n";
        std::cout << "=================================\n";

        std::cout << "Input count : " << input_count << '\n';
        std::cout << "Output count: " << output_count << '\n';

        for (size_t i = 0; i < input_count; ++i)
        {
            auto input_name =
                session.GetInputNameAllocated(i, allocator);

            auto input_type_info =
                session.GetInputTypeInfo(i);

            auto tensor_info =
                input_type_info.GetTensorTypeAndShapeInfo();

            auto shape = tensor_info.GetShape();

            std::cout << "\nInput " << i << '\n';
            std::cout << "  Name: " << input_name.get() << '\n';
            std::cout << "  Shape: ";

            for (auto dimension : shape)
            {
                std::cout << dimension << ' ';
            }

            std::cout << '\n';
        }

        for (size_t i = 0; i < output_count; ++i)
        {
            auto output_name =
                session.GetOutputNameAllocated(i, allocator);

            auto output_type_info =
                session.GetOutputTypeInfo(i);

            auto tensor_info =
                output_type_info.GetTensorTypeAndShapeInfo();

            auto shape = tensor_info.GetShape();

            std::cout << "\nOutput " << i << '\n';
            std::cout << "  Name: " << output_name.get() << '\n';
            std::cout << "  Shape: ";

            for (auto dimension : shape)
            {
                std::cout << dimension << ' ';
            }

            std::cout << '\n';
        }
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime error:\n"
                  << e.what() << '\n';

        return 1;
    }

    return 0;
}