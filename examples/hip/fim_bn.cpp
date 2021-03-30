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
#include "fim_runtime_api.h"
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "utility/fim_dump.hpp"

#ifdef DEBUG_FIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

inline float convertH2F(half h_val) { return half_float::detail::half2float<float>(h_val); }

int fim_bn_nr1(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_variance = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* golden_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_256KB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_256KB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_256KB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_256KB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_256KB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_256KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);
    // /* __FIM_API__ call : Execute FIM kernel */
    for (int i = 0; i < NUM_ITER; i++) {
        FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }
    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    //    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / 2, 0.001);

    // dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_variance);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_bn_nr2(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 2;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_variance = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* golden_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_512KB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_512KB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_512KB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_512KB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_512KB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_512KB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_512KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __FIM_API__ call : Execute FIM kernel */
        FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / 2, 0.001);

    // dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_variance);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_bn_nr3(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 4;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_variance = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* golden_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_1024KB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_1024KB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_1024KB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_1024KB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_1024KB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_1024KB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_1024KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __FIM_API__ call : Execute FIM kernel */
        FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / 2, 0.001);
    // cout <<" ITER : " << i <<endl;
    //    dump_data(preload_input.c_str(), (char*)preloaded_fim_input->data, preloaded_fim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_variance);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_bn_nr4(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 8;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_variance = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* golden_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_2048KB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_2048KB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_2048KB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_2048KB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_2048KB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_2048KB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_2048KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __FIM_API__ call : Execute FIM kernel */
        FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / 2, 0.001);
    // cout <<" ITER : " << i <<endl;
    //    dump_data(preload_input.c_str(), (char*)preloaded_fim_input->data, preloaded_fim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_variance);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_bn_nr5(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 16;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_variance = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* golden_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_4096KB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_4096KB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_4096KB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_4096KB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_4096KB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_4096KB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_4096KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __FIM_API__ call : Execute FIM kernel */
        FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / 2, 0.001);

    //    dump_data(preload_input.c_str(), (char*)preloaded_fim_input->data, preloaded_fim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_variance);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_bn_nr6(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 32;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_variance = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* golden_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_8192KB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_8192KB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_8192KB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_8192KB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_8192KB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_8192KB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_8192KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __FIM_API__ call : Execute FIM kernel */
        FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / 2, 0.001);

    //    dump_data(preload_input.c_str(), (char*)preloaded_fim_input->data, preloaded_fim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_variance);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_bn_nr7(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 64;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_variance = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* golden_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_16MB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_16MB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_16MB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_16MB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_16MB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_16MB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_16MB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __FIM_API__ call : Execute FIM kernel */
        FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / 2, 0.001);

    //    dump_data(preload_input.c_str(), (char*)preloaded_fim_input->data, preloaded_fim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_variance);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_bn_nr8(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 128;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_variance = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* golden_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_32MB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_32MB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_32MB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_32MB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_32MB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_32MB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_32MB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __FIM_API__ call : Execute FIM kernel */
        FimExecuteBN(device_output, fim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / 2, 0.001);

    //    dump_data(preload_input.c_str(), (char*)preloaded_fim_input->data, preloaded_fim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_variance);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, FimNRBN1Sync) { EXPECT_TRUE(fim_bn_nr1(true) == 0); }
TEST(HIPIntegrationTest, FimNRBN1Async) { EXPECT_TRUE(fim_bn_nr1(false) == 0); }
#if 0
TEST(HIPIntegrationTest, FimNRBN2Sync) { EXPECT_TRUE(fim_bn_nr2(true) == 0); }
TEST(HIPIntegrationTest, FimNRBN2Async) { EXPECT_TRUE(fim_bn_nr2(false) == 0); }
TEST(HIPIntegrationTest, FimNRBN3Sync) { EXPECT_TRUE(fim_bn_nr3(true) == 0); }
TEST(HIPIntegrationTest, FimNRBN3Async) { EXPECT_TRUE(fim_bn_nr3(false) == 0); }
TEST(HIPIntegrationTest, FimNRBN4Sync) { EXPECT_TRUE(fim_bn_nr4(true) == 0); }
TEST(HIPIntegrationTest, FimNRBN4Async) { EXPECT_TRUE(fim_bn_nr4(false) == 0); }
TEST(HIPIntegrationTest, FimNRBN5Sync) { EXPECT_TRUE(fim_bn_nr5(true) == 0); }
TEST(HIPIntegrationTest, FimNRBN5Async) { EXPECT_TRUE(fim_bn_nr5(false) == 0); }
TEST(HIPIntegrationTest, FimNRBN6Sync) { EXPECT_TRUE(fim_bn_nr6(true) == 0); }
TEST(HIPIntegrationTest, FimNRBN6Async) { EXPECT_TRUE(fim_bn_nr6(false) == 0); }
TEST(HIPIntegrationTest, FimNRBN7Sync) { EXPECT_TRUE(fim_bn_nr7(true) == 0); }
TEST(HIPIntegrationTest, FimNRBN7Async) { EXPECT_TRUE(fim_bn_nr7(false) == 0); }
TEST(HIPIntegrationTest, FimNRBN8Sync) { EXPECT_TRUE(fim_bn_nr8(true) == 0); }
TEST(HIPIntegrationTest, FimNRBN8Async) { EXPECT_TRUE(fim_bn_nr8(false) == 0); }
#endif
