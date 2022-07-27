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
#include "half.hpp"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_profile.h"

using half_float::half;

int pim_gemm_bias_relu(int n, int c, int inout_h, int in_w, int out_w, PimActFunc act, bool is_bias, bool block)
{
    int ret = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;
    int in_size = n * c * inout_h * in_w;
    int wei_size = n * c * in_w * out_w;
    int out_size = n * c * inout_h * out_w;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimExecuteDummy();

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimGemmDesc* pim_gemm_desc = PimCreateGemmDesc(n, c, inout_h, in_w, out_w, PIM_FP16);
    PimBo* host_input = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_INPUT);
    PimBo* host_weight = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_WEIGHT);
    PimBo* host_bias = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_BIAS);
    PimBo* host_output = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_OUTPUT);
    PimBo* device_input = PimCreateBo(pim_gemm_desc, MEM_TYPE_DEVICE, GEMM_INPUT);
    PimBo* device_weight = PimCreateBo(pim_gemm_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT);
    PimBo* device_bias = PimCreateBo(pim_gemm_desc, MEM_TYPE_DEVICE, GEMM_BIAS);
    PimBo* device_output = PimCreateBo(pim_gemm_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT);

    /* Initialize the input, weight, output data */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    set_half_data((half*)golden_output->data, half(0.0), out_size);
    set_half_data((half*)host_output->data, half(0.0), out_size);
    set_half_data((half*)host_input->data, half(dis(gen)), in_size);
    set_half_data((half*)host_weight->data, half(dis(gen)), wei_size);
    set_half_data((half*)host_bias->data, half(10.0), out_size);
    half* in_data = (half*)host_input->data;
    half* wei_data = (half*)host_weight->data;
    half* out_data = (half*)golden_output->data;
    for (int nc_i = 0; nc_i < n * c; nc_i++) {
        matmulCPU(in_data, wei_data, out_data, inout_h, out_w, in_w, half(alpha), half(beta));
        in_data += (in_w * inout_h);
        wei_data += (in_w * out_w);
        out_data += (out_w * inout_h);
    }
    if (is_bias) addBiasCPU((half*)golden_output->data, (half*)host_bias->data, out_size);
    if (act == ACT_RELU) reluCPU((half*)golden_output->data, out_size);
    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_bias, host_bias, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMM) */
    PimBo* t_device_bias = (is_bias) ? device_bias : nullptr;
    ret = PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, nullptr, block);
    if (!block) PimSynchronize();

    /* Verify result */
    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, epsilon);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_bias);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_bias);
    PimDestroyBo(device_output);
    PimDestroyGemmDesc(pim_gemm_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_fused_gemm_bias_relu(int n, int c, int inout_h, int in_w0, int out_w0, bool is_bias0, PimActFunc act0,
                             int in_w1, int out_w1, bool is_bias1, PimActFunc act1)
{
    int ret = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    int in_size0 = n * c * inout_h * in_w0;
    int wei_size0 = n * c * in_w0 * out_w0;
    int out_size0 = n * c * inout_h * out_w0;
    int in_size1 = n * c * inout_h * in_w1;
    int wei_size1 = n * c * in_w1 * out_w1;
    int out_size1 = n * c * inout_h * out_w1;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimExecuteDummy();

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimGemmDesc* pim_gemm_desc0 = PimCreateGemmDesc(n, c, inout_h, in_w0, out_w0, PIM_FP16);
    PimGemmDesc* pim_gemm_desc1 = PimCreateGemmDesc(n, c, inout_h, in_w1, out_w1, PIM_FP16);

    PimBo* host_input0 = PimCreateBo(pim_gemm_desc0, MEM_TYPE_HOST, GEMM_INPUT);
    PimBo* host_weight0 = PimCreateBo(pim_gemm_desc0, MEM_TYPE_HOST, GEMM_WEIGHT);
    PimBo* host_bias0 = PimCreateBo(pim_gemm_desc0, MEM_TYPE_HOST, GEMM_BIAS);
    PimBo* host_output0 = PimCreateBo(pim_gemm_desc0, MEM_TYPE_HOST, GEMM_OUTPUT);
    PimBo* device_input0 = PimCreateBo(pim_gemm_desc0, MEM_TYPE_DEVICE, GEMM_INPUT);
    PimBo* device_weight0 = PimCreateBo(pim_gemm_desc0, MEM_TYPE_DEVICE, GEMM_WEIGHT);
    PimBo* device_bias0 = PimCreateBo(pim_gemm_desc0, MEM_TYPE_DEVICE, GEMM_BIAS);
    PimBo* device_output0 = PimCreateBo(pim_gemm_desc0, MEM_TYPE_DEVICE, GEMM_OUTPUT);

    PimBo* host_weight1 = PimCreateBo(pim_gemm_desc1, MEM_TYPE_HOST, GEMM_WEIGHT);
    PimBo* host_bias1 = PimCreateBo(pim_gemm_desc1, MEM_TYPE_HOST, GEMM_BIAS);
    PimBo* host_output1 = PimCreateBo(pim_gemm_desc1, MEM_TYPE_HOST, GEMM_OUTPUT);
    PimBo* device_weight1 = PimCreateBo(pim_gemm_desc1, MEM_TYPE_DEVICE, GEMM_WEIGHT);
    PimBo* device_bias1 = PimCreateBo(pim_gemm_desc1, MEM_TYPE_DEVICE, GEMM_BIAS);
    PimBo* device_output1 = PimCreateBo(pim_gemm_desc1, MEM_TYPE_DEVICE, GEMM_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_gemm_desc1, MEM_TYPE_HOST, GEMM_OUTPUT);

    /* Initialize the input, weight, output data */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    set_half_data((half*)host_input0->data, half(dis(gen)), in_size0);
    set_half_data((half*)host_weight0->data, half(dis(gen)), wei_size0);
    set_half_data((half*)host_bias0->data, half(10.0), out_size0);
    set_half_data((half*)host_output0->data, half(0.0), out_size0);

    set_half_data((half*)host_weight1->data, half(dis(gen)), wei_size1);
    set_half_data((half*)host_bias1->data, half(10.0), out_size1);
    set_half_data((half*)host_output1->data, half(0.0), out_size1);
    set_half_data((half*)golden_output->data, half(0.0), out_size1);

    half* in_data0 = (half*)host_input0->data;
    half* wei_data0 = (half*)host_weight0->data;
    half* out_data0 = (half*)host_output0->data;

    half* wei_data1 = (half*)host_weight1->data;
    half* out_data1 = (half*)golden_output->data;

    for (int nc_i = 0; nc_i < n * c; nc_i++) {
        matmulCPU(in_data0, wei_data0, out_data0, inout_h, out_w0, in_w0, half(alpha), half(beta));
        in_data0 += (in_w0 * inout_h);
        wei_data0 += (in_w0 * out_w0);
        out_data0 += (out_w0 * inout_h);
    }
    if (is_bias0) addBiasCPU((half*)host_output0->data, (half*)host_bias0->data, out_size0);
    if (act0 == ACT_RELU) reluCPU((half*)host_output0->data, out_size0);

    half* in_data1 = (half*)host_output0->data;
    for (int nc_i = 0; nc_i < n * c; nc_i++) {
        matmulCPU(in_data1, wei_data1, out_data1, inout_h, out_w1, in_w1, half(alpha), half(beta));
        in_data1 += (in_w1 * inout_h);
        wei_data1 += (in_w1 * out_w1);
        out_data1 += (out_w1 * inout_h);
    }
    if (is_bias1) addBiasCPU((half*)golden_output->data, (half*)host_bias1->data, out_size1);
    if (act1 == ACT_RELU) reluCPU((half*)golden_output->data, out_size1);

    PimCopyMemory(device_input0, host_input0, HOST_TO_DEVICE);
    PimCopyMemory(device_weight0, host_weight0, HOST_TO_DEVICE);
    PimCopyMemory(device_bias0, host_bias0, HOST_TO_DEVICE);

    PimCopyMemory(device_weight1, host_weight1, HOST_TO_DEVICE);
    PimCopyMemory(device_bias1, host_bias1, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMM) */
    PimBo* t_device_bias0 = (is_bias0) ? device_bias0 : nullptr;
    PimBo* t_device_bias1 = (is_bias1) ? device_bias1 : nullptr;

    ret = PimExecuteGemm(device_output0, device_input0, device_weight0, device_bias0, act0, nullptr, false);
    ret = PimExecuteGemm(device_output1, device_output0, device_weight1, device_bias1, act1, nullptr, true);

    /* Verify result */
    PimCopyMemory(host_output1, device_output1, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output1->data, out_size1, epsilon);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input0);
    PimDestroyBo(host_weight0);
    PimDestroyBo(host_bias0);
    PimDestroyBo(host_output0);
    PimDestroyBo(device_input0);
    PimDestroyBo(device_weight0);
    PimDestroyBo(device_bias0);
    PimDestroyBo(device_output0);

    PimDestroyBo(host_weight1);
    PimDestroyBo(host_bias1);
    PimDestroyBo(host_output1);
    PimDestroyBo(device_weight1);
    PimDestroyBo(device_bias1);
    PimDestroyBo(device_output1);
    PimDestroyBo(golden_output);

    PimDestroyGemmDesc(pim_gemm_desc0);
    PimDestroyGemmDesc(pim_gemm_desc1);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemm_bias_relu_profile(int n, int c, int inout_h, int in_w, int out_w, PimActFunc act, bool is_bias, bool block)
{
    int ret = 0;
    int in_size = n * c * inout_h * in_w;
    int wei_size = n * c * in_w * out_w;
    int out_size = n * c * inout_h * out_w;
    int iter_cnt = 1000;
    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimExecuteDummy();

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimGemmDesc* pim_gemm_desc = PimCreateGemmDesc(n, c, inout_h, in_w, out_w, PIM_FP16);
    PimBo* host_input = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_INPUT);
    PimBo* host_weight = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_WEIGHT);
    PimBo* host_bias = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_BIAS);
    PimBo* host_output = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_gemm_desc, MEM_TYPE_HOST, GEMM_OUTPUT);
    PimBo* device_input = PimCreateBo(pim_gemm_desc, MEM_TYPE_DEVICE, GEMM_INPUT);
    PimBo* device_weight = PimCreateBo(pim_gemm_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT);
    PimBo* device_bias = PimCreateBo(pim_gemm_desc, MEM_TYPE_DEVICE, GEMM_BIAS);
    PimBo* device_output = PimCreateBo(pim_gemm_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT);

    /* Initialize the input, weight, output data */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    set_half_data((half*)golden_output->data, half(0.0), out_size);
    set_half_data((half*)host_output->data, half(0.0), out_size);
    set_half_data((half*)host_input->data, half(dis(gen)), in_size);
    set_half_data((half*)host_weight->data, half(dis(gen)), wei_size);
    set_half_data((half*)host_bias->data, half(10.0), out_size);
    half* in_data = (half*)host_input->data;
    half* wei_data = (half*)host_weight->data;
    half* out_data = (half*)golden_output->data;
    for (int ci = 0; ci < c; ci++) {
        matmulCPU(in_data, wei_data, out_data, inout_h, out_w, in_w, half(alpha), half(beta));
        in_data += (in_w * inout_h);
        wei_data += (in_w * out_w);
        out_data += (out_w * inout_h);
    }
    if (is_bias) addBiasCPU((half*)golden_output->data, (half*)host_bias->data, out_size);
    if (act == ACT_RELU) reluCPU((half*)golden_output->data, out_size);
    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_bias, host_bias, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* Warm up GPU and preload weight data into PIM area */
    PimBo* t_device_bias = (is_bias) ? device_bias : nullptr;
    ret = PimExecuteGemm(device_output, device_input, device_weight, t_device_bias, act, nullptr, true);

    /* Start profile */
    PIM_PROFILE_TICK_A(PimExecuteGemm);
    for (int i = 0; i < iter_cnt; i++) {
        PimExecuteGemm(device_output, device_input, device_weight, t_device_bias, act, nullptr, false);
    }
    PimSynchronize();
    PIM_PROFILE_TOCK_ITER_A(PimExecuteGemm, iter_cnt);
    /* End profile */

    /* Verify result */
    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, epsilon);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_bias);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_bias);
    PimDestroyBo(device_output);
    PimDestroyGemmDesc(pim_gemm_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, pim_gemm_bias_relu_1x1024_1024x4096)
{
    EXPECT_TRUE(pim_gemm_bias_relu(1, 1, 1, 1024, 4096, ACT_RELU, true, true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_8x1024_1024x4096)
{
    EXPECT_TRUE(pim_gemm_bias_relu(1, 1, 8, 1024, 4096, ACT_RELU, true, true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_4x8x1024_4x1024x4096)
{
    EXPECT_TRUE(pim_gemm_bias_relu(1, 4, 8, 1024, 4096, ACT_RELU, true, true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_64x1x256_64x256x64)
{
    EXPECT_TRUE(pim_gemm_bias_relu(1, 64, 1, 256, 64, ACT_RELU, true, true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_64x1x1024_64x1024x64)
{
    EXPECT_TRUE(pim_gemm_bias_relu(1, 64, 1, 1024, 64, ACT_RELU, true, true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_4x1x4096_4x4096x1024)
{
    EXPECT_TRUE(pim_gemm_bias_relu(1, 4, 1, 4096, 1024, ACT_RELU, true, true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_4x8x4096_4x4096x1024)
{
    EXPECT_TRUE(pim_gemm_bias_relu(1, 4, 8, 4096, 1024, ACT_RELU, true, true) == 0);
}
TEST(HIPIntegrationTest, pim_fused_gemm_bias_relu_4x1x1024_4x1024x4096_4x4096x1024)
{
    EXPECT_TRUE(pim_fused_gemm_bias_relu(1, 4, 1, 1024, 4096, true, ACT_RELU, 4096, 1024, true, NONE) == 0);
}
TEST(HIPIntegrationTest, pim_fused_gemm_bias_relu_4x8x1024_4x1024x4096_4x4096x1024)
{
    EXPECT_TRUE(pim_fused_gemm_bias_relu(1, 4, 8, 1024, 4096, true, ACT_RELU, 4096, 1024, true, NONE) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_4x8x1024_4x1024x4096_profile)
{
    EXPECT_TRUE(pim_gemm_bias_relu_profile(1, 4, 8, 1024, 4096, ACT_RELU, true, true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_4x8x4096_4x4096x1024_profile)
{
    EXPECT_TRUE(pim_gemm_bias_relu_profile(1, 4, 8, 4096, 1024, NONE, true, true) == 0);
}
