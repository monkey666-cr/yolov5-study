cmake_minimum_required(VERSION 3.10)

project(TensorRT_YoloV5 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

message("TensorRT SAMPLE DIR: " ${TensorRT_SAMPLE_DIR})
message("TensorRT INCLUDE DIR: " ${TensorRT_INCLUDE_DIR})
message("OpenCV INCLUDE DIRS: " ${OpenCV_INCLUDE_DIRS})
message("OpenCV LIBS DIRS: " ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

add_library(yolo_plugins
    SHARED
    src/plugins/yoloPlugins.cpp
    src/plugins/yoloForward_nc.cu
)
target_include_directories(yolo_plugins PUBLIC ${TensorRT_INCLUDE_DIR})
target_link_libraries(yolo_plugins PUBLIC ${TensorRT_LIBRARIES})
set_target_properties(yolo_plugins PROPERTIES CUDA_ARCHITECTURES "61;70;75")

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
target_link_libraries(build PRIVATE -Wl,--no-as-needed yolo_plugins) # -Wl,--no-as-needed is needed to avoid linking errors
target_link_libraries(build PRIVATE process)
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")

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
target_link_libraries(runtime PRIVATE process)
target_link_libraries(runtime PRIVATE -Wl,--no-as-needed yolo_plugins)
set_target_properties(runtime PROPERTIES CUDA_ARCHITECTURES "61;70;75")
