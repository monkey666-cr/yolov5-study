cmake_minimum_required(VERSION 3.10)

project(TensorRT_YoloV5 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

message("TensorRT SAMPLE DIR: " ${TensorRT_SAMPLE_DIR})
message("OpenCV INCLUDE DIRS: " ${OpenCV_INCLUDE_DIRS})
message("OpenCV LIBS DIRS: " ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

add_library(process
    SHARED
    src/process/postprocess.cpp
    src/process/postprocess.h
    src/process/preprocess.cu
    src/process/preprocess.h
)
target_link_libraries(process PUBLIC ${OpenCV_LIBS})
set_target_properties(process PROPERTIES CUDA_ARCHITECTURES "61;70;75")

add_executable(build
    src/build.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIR} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(build PRIVATE ${TensorRT_LIBRARIES})

add_executable(runtime
    src/runtime.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
)
target_include_directories(runtime
    PRIVATE
    ${TensorRT_INCLUDE_DIR}
    ${TensorRT_SAMPLE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/process
)
target_link_libraries(runtime PRIVATE ${TensorRT_LIBRARIES})