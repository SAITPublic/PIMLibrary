#include <assert.h>
#include <gtest/gtest.h>
#include <random>
#include "executor/gpu_hip_kernels/gpu_custom_ops.h"
#include "half.hpp"
#include "utility/pim_dump.hpp"

#define EPSILON (1.0)

using half_float::half;

void matmulCPU(half* input0, half* input1, half* output, int m, int n, int k, half alpha, half beta)
{
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            float temp = 0;
            for (int ki = 0; ki < k; ki++) {
                temp += (input0[mi * k + ki] * input1[ki * n + ni]);
            }
            int out_idx = mi * n + ni;
            output[out_idx] += alpha * temp + beta * output[out_idx];
        }
    }
}

void transposeCPU(half* in, half* out, int row, int col)
{
    for (int ri = 0; ri < row; ri++) {
        for (int ci = 0; ci < col; ci++) {
            out[ci * row + ri] = in[ri * col + ci];
        }
    }
}

int custom_gemv_Axy(bool block)
{
    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
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
    matmulCPU(input0_h, input1_h, output0_h, 1, out_size, in_size, half(alpha), half(beta));

    transposeCPU(input1_h, input1_h_t, in_size, out_size);

    hipMemcpy(input0_d, input0_h, in_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input1_d, input1_h_t, wei_bytes, hipMemcpyHostToDevice);

    rocblas_gemv_fp16_Axy(input1_d, input0_d, output_d, out_size, 1, in_size, alpha, beta);

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
    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
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
    matmulCPU(input0_h, input1_h, output0_h, 1, out_size, in_size, half(alpha), half(beta));

    hipMemcpy(input0_d, input0_h, in_bytes, hipMemcpyHostToDevice);
    hipMemcpy(input1_d, input1_h, wei_bytes, hipMemcpyHostToDevice);

    rocblas_gemv_fp16_xAy(input0_d, input1_d, output_d, 1, out_size, in_size, alpha, beta);

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

TEST(HIPIntegrationTest, CustomGemvAxyTest) { EXPECT_TRUE(custom_gemv_Axy(true) == 0); }
TEST(HIPIntegrationTest, CustomGemvxAyTest) { EXPECT_TRUE(custom_gemv_xAy(true) == 0); }
