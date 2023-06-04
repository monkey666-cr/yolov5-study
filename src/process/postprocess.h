#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>

cv::Rect get_rect(cv::Mat &img, float bbox[4]);

void yolo_nms(std::vector<Detection> &res, int32_t *num_det, int32_t *cls, float *conf, float *bbox, float conf_thresh, float nms_thresh);
