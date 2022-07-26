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

int pim_gemm_bias_relu_1x1024_1024x4096(bool block)
{
    int ret = 0;
    int n = 1;
    int c = 1;
    int inout_h = 1;
    int in_w = 1024;
    int out_w = 4096;
    PimActFunc act = ACT_RELU;
    int in_size = n * c * inout_h * in_w;
    int wei_size = n * c * in_w * out_w;
    int out_size = n * c * inout_h * out_w;

    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimGemmDesc* pim_gemm_desc = PimCreateGemmDesc(n, c, inout_h, in_w, out_w, PIM_FP16);
    /* __PIM_API__ call : Create PIM Buffer Object */
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
    set_half_data((half*)golden_output->data, half(0.0), out_size);
    set_half_data((half*)host_output->data, half(0.0), out_size);
    set_half_data((half*)host_input->data, half(dis(gen)), in_size);
    set_half_data((half*)host_weight->data, half(dis(gen)), wei_size);
    set_half_data((half*)host_bias->data, half(10.0), out_size);

    matmulCPU((half*)host_input->data, (half*)host_weight->data, (half*)golden_output->data, 1, out_size, in_size,
              half(alpha), half(beta));
    addBiasCPU((half*)golden_output->data, (half*)host_bias->data, out_size);
    if (act == ACT_RELU) reluCPU((half*)golden_output->data, out_size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_bias, host_bias, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMM) */
    ret = PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, nullptr, block);
    if (!block) PimSynchronize();

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

int pim_gemm_bias_relu_8x1024_1024x4096(bool block)
{
    int ret = 0;
    int n = 1;
    int c = 1;
    int inout_h = 8;
    int in_w = 1024;
    int out_w = 4096;
    PimActFunc act = ACT_RELU;
    int in_size = n * c * inout_h * in_w;
    int wei_size = n * c * in_w * out_w;
    int out_size = n * c * inout_h * out_w;

    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimGemmDesc* pim_gemm_desc = PimCreateGemmDesc(n, c, inout_h, in_w, out_w, PIM_FP16);
    /* __PIM_API__ call : Create PIM Buffer Object */
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
    set_half_data((half*)golden_output->data, half(0.0), out_size);
    set_half_data((half*)host_output->data, half(0.0), out_size);
    set_half_data((half*)host_input->data, half(dis(gen)), in_size);
    set_half_data((half*)host_weight->data, half(dis(gen)), wei_size);
    set_half_data((half*)host_bias->data, half(10.0), out_size);

    matmulCPU((half*)host_input->data, (half*)host_weight->data, (half*)golden_output->data, inout_h, out_w, in_w,
              half(alpha), half(beta));
    addBiasCPU((half*)golden_output->data, (half*)host_bias->data, out_size);
    if (act == ACT_RELU) reluCPU((half*)golden_output->data, out_size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_bias, host_bias, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMM) */
    ret = PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, nullptr, block);
    if (!block) PimSynchronize();

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

int pim_gemm_bias_relu_4x8x1024_4x1024x4096(bool block)
{
    int ret = 0;
    int n = 1;
    int c = 4;
    int inout_h = 8;
    int in_w = 1024;
    int out_w = 4096;
    PimActFunc act = ACT_RELU;
    int in_size = n * c * inout_h * in_w;
    int wei_size = n * c * in_w * out_w;
    int out_size = n * c * inout_h * out_w;

    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimGemmDesc* pim_gemm_desc = PimCreateGemmDesc(n, c, inout_h, in_w, out_w, PIM_FP16);
    /* __PIM_API__ call : Create PIM Buffer Object */
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
    addBiasCPU((half*)golden_output->data, (half*)host_bias->data, out_size);
    if (act == ACT_RELU) reluCPU((half*)golden_output->data, out_size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_bias, host_bias, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMM) */
    ret = PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, nullptr, block);
    if (!block) PimSynchronize();

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

int pim_gemm_bias_relu_64x1x256_64x256x64(bool block)
{
    int ret = 0;
    int n = 1;
    int c = 64;
    int inout_h = 1;
    int in_w = 256;
    int out_w = 64;
    PimActFunc act = ACT_RELU;
    int in_size = n * c * inout_h * in_w;
    int wei_size = n * c * in_w * out_w;
    int out_size = n * c * inout_h * out_w;

    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimGemmDesc* pim_gemm_desc = PimCreateGemmDesc(n, c, inout_h, in_w, out_w, PIM_FP16);
    /* __PIM_API__ call : Create PIM Buffer Object */
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
    addBiasCPU((half*)golden_output->data, (half*)host_bias->data, out_size);
    if (act == ACT_RELU) reluCPU((half*)golden_output->data, out_size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_bias, host_bias, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMM) */
    ret = PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, nullptr, block);
    if (!block) PimSynchronize();

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

int pim_gemm_bias_relu_64x1x1024_64x1024x64(bool block)
{
    int ret = 0;
    int n = 1;
    int c = 64;
    int inout_h = 1;
    int in_w = 1024;
    int out_w = 64;
    PimActFunc act = ACT_RELU;
    int in_size = n * c * inout_h * in_w;
    int wei_size = n * c * in_w * out_w;
    int out_size = n * c * inout_h * out_w;

    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimGemmDesc* pim_gemm_desc = PimCreateGemmDesc(n, c, inout_h, in_w, out_w, PIM_FP16);
    /* __PIM_API__ call : Create PIM Buffer Object */
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
    addBiasCPU((half*)golden_output->data, (half*)host_bias->data, out_size);
    if (act == ACT_RELU) reluCPU((half*)golden_output->data, out_size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_bias, host_bias, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMM) */
    ret = PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, nullptr, block);
    if (!block) PimSynchronize();

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

int pim_gemm_bias_relu_4x1x4096_4x4096x1024(bool block)
{
    int ret = 0;
    int n = 1;
    int c = 4;
    int inout_h = 1;
    int in_w = 4096;
    int out_w = 1024;
    PimActFunc act = ACT_RELU;
    int in_size = n * c * inout_h * in_w;
    int wei_size = n * c * in_w * out_w;
    int out_size = n * c * inout_h * out_w;

    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimGemmDesc* pim_gemm_desc = PimCreateGemmDesc(n, c, inout_h, in_w, out_w, PIM_FP16);
    /* __PIM_API__ call : Create PIM Buffer Object */
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
    addBiasCPU((half*)golden_output->data, (half*)host_bias->data, out_size);
    if (act == ACT_RELU) reluCPU((half*)golden_output->data, out_size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_bias, host_bias, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMM) */
    ret = PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, nullptr, block);
    if (!block) PimSynchronize();

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
    EXPECT_TRUE(pim_gemm_bias_relu_1x1024_1024x4096(true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_8x1024_1024x4096)
{
    EXPECT_TRUE(pim_gemm_bias_relu_8x1024_1024x4096(true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_4x8x1024_4x1024x4096)
{
    EXPECT_TRUE(pim_gemm_bias_relu_4x8x1024_4x1024x4096(true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_64x1x256_64x256x64)
{
    EXPECT_TRUE(pim_gemm_bias_relu_64x1x256_64x256x64(true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_64x1x1024_64x1024x64)
{
    EXPECT_TRUE(pim_gemm_bias_relu_64x1x1024_64x1024x64(true) == 0);
}
TEST(HIPIntegrationTest, pim_gemm_bias_relu_4x1x4096_4x4096x1024)
{
    EXPECT_TRUE(pim_gemm_bias_relu_4x1x4096_4x4096x1024(true) == 0);
}
