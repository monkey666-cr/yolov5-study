#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "cassert"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "用法: ./build [onnx_file_path]" << std::endl;
        return -1;
    }
    // 命令行中的onnx文件路径
    char *onnx_file_path = argv[1];

    std::cout << "onnx file path: " << onnx_file_path << std::endl;

    // ====== 1. 创建 builder ======
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        std::cerr << "Failed to create builder" << std::endl;

        return -1;
    }

    // ====== 2. 创建 network: builder ---> network ======
    // 显性 batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 调用builder的createNetworkV2方法创建network
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cerr << "Failed to create network" << std::endl;

        return -1;
    }

    // 使用 onnxparser 创建网络
    // 创建 onnxparser, 用于解析onnx文件
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    // 调用 onnxparser 的 parseFromFile 解析 onnx 文件
    bool parsed = parser->parseFromFile(onnx_file_path, static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        std::cerr << "Failed to parse onnx file" << std::endl;

        return -1;
    }
    // 配置网络参数
    // 告诉 tensorrt 运行时, 输入图像的范围, batch size 范围, 这样 tensorrt 才能对模型和进行构建和优化
    auto input = network->getInput(0);                                                                             // 获取输入节点
    auto profile = builder->createOptimizationProfile();                                                           // 创建profile, 用于设置输入的动态尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 640, 640}); // 设置最小尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 640, 640}); // 设置最优尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 640, 640}); // 设置最大尺寸

    // ====== 3. 创建config配置: builder ---> config ======
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << "Failed to create config" << std::endl;

        return -1;
    }
    // 使用 addOptimizationProfile 方法添加 profile, 用于输入的动态尺寸
    config->addOptimizationProfile(profile);
    // 设置精度, 不设置是FP32, 设置位FP16, 设置位INT8需要额外设置calibrator
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // 设置最大batchsize
    builder->setMaxBatchSize(1);
    // 设置最大工作空间axBatchSize(int32_t)’ is deprecated 
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

    // 创建流, 用于设置 profile
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        std::cerr << "Failed to create CUDA stream" << std::endl;

        return -1;
    }
    config->setProfileStream(*profileStream);

    // ====== 4. 创建 engine: builder ---> engine(*nework, *config) ======
    // 使用buildSerializeNetwork方法创建engine, 可直接返回序列化的engine (原来的buildEngineWithConfig方法已经废弃，需要先创建engine，再序列化)
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        std::cout << "failed to create engine" << std::endl;

        return -1;
    }

    // ====== 5. 序列化保存 engine ======
    std::ofstream engine_file("./weights/yolov5.engine", std::ios::binary);
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    // ====== 6. 释放资源 ======
    // 智能指针会自动释放

    std::cout << "Engine build success!" << std::endl;

    return 0;
}
