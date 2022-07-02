
/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <random>
// #include "half.hpp"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

#define IN_LENGTH 256

using namespace std;

template <class T>
void fill_uniform_random_values(void* data, uint32_t count, T start, T end)
{
    std::random_device rd;
    std::mt19937 mt(rd());

    std::uniform_real_distribution<double> dist(start, end);

    for (int i = 0; i < count; i++) ((T*)data)[i] = dist(mt);
}

void calculate_add(half_float::half* inp1, half_float::half* inp2, half_float::half* output)
{
    for (int i = 0; i < IN_LENGTH; i++) {
        output[i] = inp1[i] + inp2[i];
    }
}

bool gpu_add()
{
    int ret = 0;

    PimInitialize(RT_TYPE_OPENCL, PIM_FP16);

    PimBo* host_opr1 = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_opr2 = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_out = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* ref_out = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_opr1 = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_opr2 = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);

    fill_uniform_random_values<half_float::half>(host_opr1->data, IN_LENGTH, (half_float::half)0.0,
                                                 (half_float::half)0.5);
    fill_uniform_random_values<half_float::half>(host_opr2->data, IN_LENGTH, (half_float::half)0.0,
                                                 (half_float::half)0.5);

    calculate_add((half_float::half*)host_opr1->data, (half_float::half*)host_opr2->data,
                  (half_float::half*)ref_out->data);
    PimCopyMemory(device_opr1, host_opr1, HOST_TO_DEVICE);
    PimCopyMemory(device_opr2, host_opr2, HOST_TO_DEVICE);

    PimExecuteAdd(device_output, device_opr1, device_opr2, nullptr, false);
    PimCopyMemory(host_out, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half_float::half*)host_out->data, (half_float::half*)ref_out->data, IN_LENGTH);
    if (ret != 0) {
        return false;
    }
    return true;
}

TEST(UnitTest, gpuAdd) { EXPECT_TRUE(gpu_add()); }
