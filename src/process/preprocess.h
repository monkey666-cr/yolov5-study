#pragma once

#include <opencv2/opencv.hpp>

void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height);

void process_input(cv::Mat &input_img, float *input_device_buffer);

void process_input_cv_affine(cv::Mat &src, float *input_device_buffer);

void process_input_cpu(cv::Mat &src, float *input_device_buffer);
