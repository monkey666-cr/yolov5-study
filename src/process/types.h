#pragma once

#include "config.h"

struct YoloKernel
{
    int width;
    int height;
    float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection
{
    float bbox[4];
    float conf;
    float class_id;
};
