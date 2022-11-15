/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
*/
#include <assert.h>
#include <gtest/gtest.h>
#include <random>
#include "executor/hip/gpu_custom_ops.h"
#include "half.hpp"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"

#define EPSILON (1.0)

using half_float::half;

int custom_gemv_Axy(bool block)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    int ret = 0;
    int in_size = 320;
    int out_size = 1280;
    int in_bytes = in_size * sizeof(half);
    int wei_bytes = in_size * out_size * sizeof(half);
    int out_bytes = out_size * sizeof(half);

    float alpha = 1.0f;
    float beta = 0.0f;

    half* input0_h = (half*)malloc(in_bytes);
    half* input1_h = (half*)malloc(wei_bytes);
    half* input1_h_t = (half*)malloc(wei_bytes);
    half* output0_h = (half*)malloc(out_bytes);
    half* output1_h = (half*)malloc(out_bytes);

    half* input0_d;
    half* input1_d;
    half* output_d;

    hipMalloc(&input0_d, in_bytes);
    hipMalloc(&input1_d, wei_bytes);
    hipMalloc(&output_d, out_bytes);

    for (int i = 0; i < in_size; i++) {
        input0_h[i] = half(dis(gen));
    }

    for (int i = 0; i < in_size * out_size; i++) {
        input1_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        output0_h[i] = half(0.0);
        output1_h[i] = half(0.0);
    }

    matmulCPU(input0_h, input1_h, output0_h, 1, out_size, in_size, half(alpha), half(beta));

    transposeCPU(input1_h, input1_h_t, in_size, out_size);

    hipMemcpy(input0_d, input0_h, in_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input1_d, input1_h_t, wei_bytes, hipMemcpyHostToDevice);

    rocblas_gemv_fp16_Axy(input1_d, input0_d, output_d, out_size, 1, in_size, alpha, beta, 0);

    hipMemcpy(output1_h, output_d, out_bytes, hipMemcpyDeviceToHost);

    ret = compare_half_relative(output0_h, output1_h, out_size, EPSILON);

    hipFree(input0_d);
    hipFree(input1_d);
    hipFree(output_d);
    free(input0_h);
    free(input1_h);
    free(input1_h_t);
    free(output0_h);
    free(output1_h);

    return ret;
}

int custom_gemv_xAy(bool block)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    int ret = 0;
    int in_size = 320;
    int out_size = 1280;
    int in_bytes = in_size * sizeof(half);
    int wei_bytes = in_size * out_size * sizeof(half);
    int out_bytes = out_size * sizeof(half);

    float alpha = 1.0f;
    float beta = 0.0f;

    half* input0_h = (half*)malloc(in_bytes);
    half* input1_h = (half*)malloc(wei_bytes);
    half* output0_h = (half*)malloc(out_bytes);
    half* output1_h = (half*)malloc(out_bytes);

    half* input0_d;
    half* input1_d;
    half* output_d;

    hipMalloc(&input0_d, in_bytes);
    hipMalloc(&input1_d, wei_bytes);
    hipMalloc(&output_d, out_bytes);

    for (int i = 0; i < in_size; i++) {
        input0_h[i] = half(dis(gen));
    }

    for (int i = 0; i < in_size * out_size; i++) {
        input1_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        output0_h[i] = half(0.0);
        output1_h[i] = half(0.0);
    }

    matmulCPU(input0_h, input1_h, output0_h, 1, out_size, in_size, half(alpha), half(beta));

    hipMemcpy(input0_d, input0_h, in_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input1_d, input1_h, wei_bytes, hipMemcpyHostToDevice);

    rocblas_gemv_fp16_xAy(input0_d, input1_d, output_d, 1, out_size, in_size, alpha, beta, 0);

    hipMemcpy(output1_h, output_d, out_bytes, hipMemcpyDeviceToHost);

    ret = compare_half_relative(output0_h, output1_h, out_size, EPSILON);

    hipFree(input0_d);
    hipFree(input1_d);
    hipFree(output_d);
    free(input0_h);
    free(input1_h);
    free(output0_h);
    free(output1_h);

    return ret;
}

int custom_addmv_Axy(bool relu)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    int ret = 0;
    int in_size = 320;
    int out_size = 1280;
    int in_bytes = in_size * sizeof(half);
    int wei_bytes = in_size * out_size * sizeof(half);
    int out_bytes = out_size * sizeof(half);

    float alpha = 1.0f;
    float beta = 0.0f;

    half* input0_h = (half*)malloc(in_bytes);
    half* input1_h = (half*)malloc(wei_bytes);
    half* input1_h_t = (half*)malloc(wei_bytes);
    half* input2_h = (half*)malloc(out_bytes);
    half* output0_h = (half*)malloc(out_bytes);
    half* output1_h = (half*)malloc(out_bytes);

    half* input0_d;
    half* input1_d;
    half* input2_d;
    half* output_d;

    hipMalloc(&input0_d, in_bytes);
    hipMalloc(&input1_d, wei_bytes);
    hipMalloc(&input2_d, out_bytes);
    hipMalloc(&output_d, out_bytes);

    for (int i = 0; i < in_size; i++) {
        input0_h[i] = half(dis(gen));
    }

    for (int i = 0; i < in_size * out_size; i++) {
        input1_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        input2_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        output0_h[i] = half(0.0);
        output1_h[i] = half(0.0);
    }

    matmulCPU(input0_h, input1_h, output0_h, 1, out_size, in_size, half(alpha), half(beta));

    for (int i = 0; i < out_size; i++) {
        output0_h[i] += input2_h[i];
    }

    if (relu) {
        for (int i = 0; i < out_size; i++) {
            output0_h[i] = output0_h[i] > 0.f ? output0_h[i] : 0.f;
        }
    }

    transposeCPU(input1_h, input1_h_t, in_size, out_size);

    hipMemcpy(input0_d, input0_h, in_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input1_d, input1_h_t, wei_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input2_d, input2_h, out_bytes, hipMemcpyHostToDevice);

    rocblas_addmv_fp16_Axy(input2_d, input1_d, input0_d, output_d, out_size, 1, in_size, alpha, beta, relu, 0);

    hipMemcpy(output1_h, output_d, out_bytes, hipMemcpyDeviceToHost);

    ret = compare_half_relative(output0_h, output1_h, out_size, EPSILON);

    hipFree(input0_d);
    hipFree(input1_d);
    hipFree(input2_d);
    hipFree(output_d);
    free(input0_h);
    free(input1_h);
    free(input1_h_t);
    free(input2_h);
    free(output0_h);
    free(output1_h);

    return ret;
}

int custom_addmv_xAy(bool relu)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    int ret = 0;
    int in_size = 320;
    int out_size = 1280;
    int in_bytes = in_size * sizeof(half);
    int wei_bytes = in_size * out_size * sizeof(half);
    int out_bytes = out_size * sizeof(half);

    float alpha = 1.0f;
    float beta = 0.0f;

    half* input0_h = (half*)malloc(in_bytes);
    half* input1_h = (half*)malloc(wei_bytes);
    half* input2_h = (half*)malloc(out_bytes);
    half* output0_h = (half*)malloc(out_bytes);
    half* output1_h = (half*)malloc(out_bytes);

    half* input0_d;
    half* input1_d;
    half* input2_d;
    half* output_d;

    hipMalloc(&input0_d, in_bytes);
    hipMalloc(&input1_d, wei_bytes);
    hipMalloc(&input2_d, out_bytes);
    hipMalloc(&output_d, out_bytes);

    for (int i = 0; i < in_size; i++) {
        input0_h[i] = half(dis(gen));
    }

    for (int i = 0; i < in_size * out_size; i++) {
        input1_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        input2_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        output0_h[i] = half(0.0);
        output1_h[i] = half(0.0);
    }

    matmulCPU(input0_h, input1_h, output0_h, 1, out_size, in_size, half(alpha), half(beta));

    for (int i = 0; i < out_size; i++) {
        output0_h[i] += input2_h[i];
    }

    if (relu) {
        for (int i = 0; i < out_size; i++) {
            output0_h[i] = output0_h[i] > 0.f ? output0_h[i] : 0.f;
        }
    }

    hipMemcpy(input0_d, input0_h, in_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input1_d, input1_h, wei_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input2_d, input2_h, out_bytes, hipMemcpyHostToDevice);

    rocblas_addmv_fp16_xAy(input2_d, input0_d, input1_d, output_d, 1, out_size, in_size, alpha, beta, relu, 0);

    hipMemcpy(output1_h, output_d, out_bytes, hipMemcpyDeviceToHost);

    ret = compare_half_relative(output0_h, output1_h, out_size, EPSILON);

    hipFree(input0_d);
    hipFree(input1_d);
    hipFree(input2_d);
    hipFree(output_d);
    free(input0_h);
    free(input1_h);
    free(input2_h);
    free(output0_h);
    free(output1_h);

    return ret;
}

int custom_gemv_Axy_api(bool block)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    int ret = 0;
    int in_size = 320;
    int out_size = 1280;
    int in_bytes = in_size * sizeof(half);
    int wei_bytes = in_size * out_size * sizeof(half);
    int out_bytes = out_size * sizeof(half);

    float alpha = 1.0f;
    float beta = 0.0f;

    half* input0_h = (half*)malloc(in_bytes);
    half* input1_h = (half*)malloc(wei_bytes);
    half* input1_h_t = (half*)malloc(wei_bytes);
    half* output0_h = (half*)malloc(out_bytes);
    half* output1_h = (half*)malloc(out_bytes);

    half* input0_d;
    half* input1_d;
    half* output_d;

    hipMalloc(&input0_d, in_bytes);
    hipMalloc(&input1_d, wei_bytes);
    hipMalloc(&output_d, out_bytes);

    for (int i = 0; i < in_size; i++) {
        input0_h[i] = half(dis(gen));
    }

    for (int i = 0; i < in_size * out_size; i++) {
        input1_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        output0_h[i] = half(0.0);
        output1_h[i] = half(0.0);
    }

    matmulCPU(input0_h, input1_h, output0_h, 1, out_size, in_size, half(alpha), half(beta));

    transposeCPU(input1_h, input1_h_t, in_size, out_size);

    hipMemcpy(input0_d, input0_h, in_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input1_d, input1_h_t, wei_bytes, hipMemcpyHostToDevice);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimGemmDesc* pim_desc = PimCreateGemmDesc(1, 1, in_size, 1, out_size, 1, PIM_FP16, W_X_I);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, input0_d);
    PimBo* device_weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, input1_d);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, output_d);

    /* __PIM_API__ call : Execute GEMM */
    PimExecuteGemm(device_output, device_input, device_weight, nullptr, PimActFunc::NONE, W_X_I, nullptr, block);

    hipMemcpy(output1_h, device_output->data, out_bytes, hipMemcpyDeviceToHost);

    ret = compare_half_relative(output0_h, output1_h, out_size, EPSILON);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);
    PimDestroyGemmDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    hipFree(input0_d);
    hipFree(input1_d);
    hipFree(output_d);
    free(input0_h);
    free(input1_h);
    free(input1_h_t);
    free(output0_h);
    free(output1_h);

    return ret;
}

int custom_addmv_Axy_api(bool relu)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    int ret = 0;
    int in_size = 320;
    int out_size = 1280;
    int in_bytes = in_size * sizeof(half);
    int wei_bytes = in_size * out_size * sizeof(half);
    int out_bytes = out_size * sizeof(half);
    PimActFunc act = PimActFunc::NONE;

    float alpha = 1.0f;
    float beta = 0.0f;

    half* input0_h = (half*)malloc(in_bytes);
    half* input1_h = (half*)malloc(wei_bytes);
    half* input1_h_t = (half*)malloc(wei_bytes);
    half* input2_h = (half*)malloc(out_bytes);
    half* output0_h = (half*)malloc(out_bytes);
    half* output1_h = (half*)malloc(out_bytes);

    half* input0_d;
    half* input1_d;
    half* input2_d;
    half* output_d;

    hipMalloc(&input0_d, in_bytes);
    hipMalloc(&input1_d, wei_bytes);
    hipMalloc(&input2_d, out_bytes);
    hipMalloc(&output_d, out_bytes);

    for (int i = 0; i < in_size; i++) {
        input0_h[i] = half(dis(gen));
    }

    for (int i = 0; i < in_size * out_size; i++) {
        input1_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        input2_h[i] = half(dis(gen));
    }

    for (int i = 0; i < out_size; i++) {
        output0_h[i] = half(0.0);
        output1_h[i] = half(0.0);
    }

    matmulCPU(input0_h, input1_h, output0_h, 1, out_size, in_size, half(alpha), half(beta));

    for (int i = 0; i < out_size; i++) {
        output0_h[i] += input2_h[i];
    }

    if (relu) {
        for (int i = 0; i < out_size; i++) {
            output0_h[i] = output0_h[i] > 0.f ? output0_h[i] : 0.f;
        }
        act = PimActFunc::ACT_RELU;
    }

    transposeCPU(input1_h, input1_h_t, in_size, out_size);

    hipMemcpy(input0_d, input0_h, in_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input1_d, input1_h_t, wei_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input2_d, input2_h, out_bytes, hipMemcpyHostToDevice);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimGemmDesc* pim_desc = PimCreateGemmDesc(1, 1, in_size, 1, out_size, 1, PIM_FP16, W_X_I);
    PimBo* device_vec = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, input0_d);
    PimBo* device_mat = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, input1_d);
    PimBo* device_in = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_BIAS, input2_d);
    PimBo* device_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, output_d);

    /* __PIM_API__ call : Execute GEMM ADD*/
    PimExecuteGemm(device_out, device_vec, device_mat, device_in, act, W_X_I, nullptr, false);

    hipMemcpy(output1_h, device_out->data, out_bytes, hipMemcpyDeviceToHost);

    ret = compare_half_relative(output0_h, output1_h, out_size, EPSILON);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(device_in);
    PimDestroyBo(device_vec);
    PimDestroyBo(device_mat);
    PimDestroyBo(device_out);
    PimDestroyGemmDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    hipFree(input0_d);
    hipFree(input1_d);
    hipFree(input2_d);
    hipFree(output_d);
    free(input0_h);
    free(input1_h);
    free(input1_h_t);
    free(input2_h);
    free(output0_h);
    free(output1_h);

    return ret;
}

TEST(HIPIntegrationTest, CustomGemvAxyTest) { EXPECT_TRUE(custom_gemv_Axy(true) == 0); }
TEST(HIPIntegrationTest, CustomGemvxAyTest) { EXPECT_TRUE(custom_gemv_xAy(true) == 0); }
TEST(HIPIntegrationTest, CustomAddmvAxyTest) { EXPECT_TRUE(custom_addmv_Axy(false) == 0); }
TEST(HIPIntegrationTest, CustomAddmvAxyReluTest) { EXPECT_TRUE(custom_addmv_Axy(true) == 0); }
TEST(HIPIntegrationTest, CustomAddmvxAyTest) { EXPECT_TRUE(custom_addmv_xAy(false) == 0); }
TEST(HIPIntegrationTest, CustomAddmvxAyReluTest) { EXPECT_TRUE(custom_addmv_xAy(true) == 0); }
TEST(HIPIntegrationTest, CustomGemvAxyAPITest) { EXPECT_TRUE(custom_gemv_Axy_api(false) == 0); }
TEST(HIPIntegrationTest, CustomAddmvAxyAPITest) { EXPECT_TRUE(custom_addmv_Axy_api(true) == 0); }
