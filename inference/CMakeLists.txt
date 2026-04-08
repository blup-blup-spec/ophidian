cmake_minimum_required(VERSION 3.10)
project(snake_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ARM-specific optimizations for Raspberry Pi 4
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_options(-march=armv8-a+crc+crypto -O3 -ffast-math)
    message(STATUS "Building for ARM64 (Raspberry Pi 4)")
endif()

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Find NCNN
# Try system-installed ncnn first, then fallback to custom path
find_package(ncnn QUIET)
if(NOT ncnn_FOUND)
    # If ncnn is installed via apt or built manually
    if(EXISTS "/usr/local/lib/cmake/ncnn")
        set(ncnn_DIR "/usr/local/lib/cmake/ncnn")
    elseif(EXISTS "/opt/ncnn/build/install/lib/cmake/ncnn")
        set(ncnn_DIR "/opt/ncnn/build/install/lib/cmake/ncnn")
    endif()
    find_package(ncnn REQUIRED)
endif()
message(STATUS "NCNN found: ${ncnn_DIR}")

# Build the snake detector executable
add_executable(snake_detector
    main.cpp
    snake_detector.cpp
    snake_detector.h
)

target_link_libraries(snake_detector
    ${OpenCV_LIBS}
    ncnn
)

# Enable OpenMP for multi-threaded inference
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(snake_detector OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — multi-threaded inference enabled")
endif()

# Install
install(TARGETS snake_detector DESTINATION bin)

message(STATUS "")
message(STATUS "=== Snake Detector Build Config ===")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Processor:     ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  OpenCV:        ${OpenCV_VERSION}")
message(STATUS "  NCNN:          ${ncnn_DIR}")
message(STATUS "===================================")
