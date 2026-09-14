cmake_minimum_required(VERSION 3.16)

project(driver_monitoring_system)

if(APPLE)
    set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv5")
endif()
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "/opt/homebrew/opt/onnxruntime")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(driver_monitoring_system
    src/main.cpp
    src/performance_monitor.cpp
    src/event_logger.cpp
    src/utils.cpp
    src/eye_monitor.cpp
    src/face_landmark_model.cpp
)

target_include_directories(driver_monitoring_system
    PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(driver_monitoring_system
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_videoio
        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib
)