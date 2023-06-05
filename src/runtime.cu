#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "common.h"
#include "buffers.h"

#include "process/types.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

std::vector<unsigned char> load_engine_file(const std::string &file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);

    assert(engine_file.is_open() && "Unable to load engine file");

    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();

    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);

    return engine_data;
}

int main(int argc, char **argv)
{
    /* code */
    if (argc < 3)
    {
        std::cerr << "用法: " << argv[0] << "<engine_file> <input_path_file>" << std::endl;

        return -1;
    }

    auto engine_file = argv[1];
    auto input_video_path = argv[2];

    // ====== 1. 创建推理运行时 runtime ======
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        std::cerr << "runtime create failed" << std::endl;

        return -1;
    }

    // ====== 2. 反序列化生成 engine ======
    // 加载模型文件
    auto plan = load_engine_file(engine_file);
    // 反序列化生成engine
    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!mEngine)
    {
        std::cerr << "Deserialize CUDA Engine Failed" << std::endl;

        return -1;
    }

    // ====== 3. 创建执行上下文 context ======
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "context create failed" << std::endl;

        return -1;
    }

    // ====== 4. 创建输入输出缓冲区 ======
    samplesCommon::BufferManager buffers(mEngine);

    auto cap = cv::VideoCapture(input_video_path);

    int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = int(cap.get(cv::CAP_PROP_FPS));

    // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
    cv::VideoWriter writer("./output/record.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));

    cv::Mat frame;
    int frame_index{0};

    // 申请GPU内存
    cuda_preprocess_init(width * height);

    while (cap.isOpened())
    {
        // 统计运行时间
        auto start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "文件处理完毕" << std::endl;
            break;
        }
        frame_index++;
        // 输入预处理
        process_input(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));

        // ====== 5. 执行推理 ======
        context->executeV2(buffers.getDeviceBindings().data());

        // 拷贝回host
        buffers.copyOutputToHost();

        // 从buffer manager中获取模型输出
        int32_t *num_det = (int32_t *)buffers.getHostBuffer(kOutNumDet);
        int32_t *cls = (int32_t *)buffers.getHostBuffer(kOutDetCls);
        float *conf = (float *)buffers.getHostBuffer(kOutDetScores);
        float *bbox = (float *)buffers.getHostBuffer(kOutDetBBoxes);

        std::vector<Detection> bboxs;
        yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);
        
        // 结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto time_str = std::to_string(elapsed) + "ms";
        auto fps_str = std::to_string(1000 / elapsed) + "fps";


        // 遍历检测结果
        for (size_t j = 0; j < bboxs.size(); j++)
        {
            cv::Rect r = get_rect(frame, bboxs[j].bbox);
            cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(frame, std::to_string((int)bboxs[j].class_id), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
        }

        cv::putText(frame, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        cv::putText(frame, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

        cv::imshow("frame", frame);
        // 写入视频文件
        writer.write(frame);
        std::cout << "处理完第" << frame_index << "帧" << std::endl;
        if (cv::waitKey(1) == 27)
            break;
    }
    
    // ====== 6. 释放内存 ======
    cuda_preprocess_destroy();

    return 0;
}
