#include "cuda.h"
#include "config.h"
#include "preprocess.h"

static uint8_t *img_buffer_device = nullptr;


void cuda_preprocess_init(int max_image_size)
{
    // 申请设备内存
    CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy()
{
    CUDA_CHECK(cudaFree(img_buffer_device));
}

void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height)
{
    int img_size = src_width * src_height * 3;
    CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice))
}

void process_input(cv::Mat &src, float *input_device_buffer)
{
    cuda_preprocess(src.ptr(), src.cols, src.rows, input_device_buffer, kInputW, kInputH);
}