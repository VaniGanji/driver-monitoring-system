#include <iostream>

#include <onnxruntime/onnxruntime_cxx_api.h>

int main()
{
    std::cout << "=================================\n";
    std::cout << "Driver Monitoring System\n";
    std::cout << "Modern C++ Reference Implementation\n";
    std::cout << "Version: 2.0\n";
    std::cout << "=================================\n";

    std::cout << "ONNX Runtime smoke test..." << std::endl;

    try
    {
        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "DMS_ONNX_SMOKE_TEST"
        );

        std::cout << "ONNX Runtime environment created successfully."
                  << std::endl;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime error: "
                  << e.what()
                  << std::endl;

        return 1;
    }

    std::cout << "ONNX Runtime smoke test PASSED." << std::endl;

    return 0;
}