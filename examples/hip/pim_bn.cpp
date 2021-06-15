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
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "pim_runtime_api.h"
#include "utility/pim_dump.hpp"

#ifdef DEBUG_PIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

inline float convertH2F(half h_val) { return half_float::detail::half2float<float>(h_val); }

int pim_bn_nr1(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

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

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);

    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);
    // /* __PIM_API__ call : Execute PIM kernel */
    for (int i = 0; i < NUM_ITER; i++) {
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }
    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    //    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / 2);

    // dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_bn_nr2(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 2;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

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

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);

    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __PIM_API__ call : Execute PIM kernel */
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    // printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / 2);

    // dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_bn_nr3(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 4;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

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

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);

    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __PIM_API__ call : Execute PIM kernel */
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    // printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / 2);
    // cout <<" ITER : " << i <<endl;
    //    dump_data(preload_input.c_str(), (char*)preloaded_pim_input->data, preloaded_pim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_bn_nr4(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 8;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

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

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);

    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __PIM_API__ call : Execute PIM kernel */
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / 2);
    // cout <<" ITER : " << i <<endl;
    //    dump_data(preload_input.c_str(), (char*)preloaded_pim_input->data, preloaded_pim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_bn_nr5(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 16;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

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

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);

    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __PIM_API__ call : Execute PIM kernel */
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / 2);

    //    dump_data(preload_input.c_str(), (char*)preloaded_pim_input->data, preloaded_pim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_bn_nr6(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 32;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

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

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);

    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __PIM_API__ call : Execute PIM kernel */
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / 2);

    //    dump_data(preload_input.c_str(), (char*)preloaded_pim_input->data, preloaded_pim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_bn_nr7(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 64;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

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

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);

    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __PIM_API__ call : Execute PIM kernel */
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / 2);

    //    dump_data(preload_input.c_str(), (char*)preloaded_pim_input->data, preloaded_pim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_bn_nr8(bool block)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 131072 * 128;
    const int HEIGHT = 1;
    const int PARAMS = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

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

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);

    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);

    for (int i = 0; i < NUM_ITER; i++) {
        // /* __PIM_API__ call : Execute PIM kernel */
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / 2);

    //    dump_data(preload_input.c_str(), (char*)preloaded_pim_input->data, preloaded_pim_input->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, PimNRBN1Sync) { EXPECT_TRUE(pim_bn_nr1(true) == 0); }
TEST(HIPIntegrationTest, PimNRBN1Async) { EXPECT_TRUE(pim_bn_nr1(false) == 0); }
#if 0
TEST(HIPIntegrationTest, PimNRBN2Sync) { EXPECT_TRUE(pim_bn_nr2(true) == 0); }
TEST(HIPIntegrationTest, PimNRBN2Async) { EXPECT_TRUE(pim_bn_nr2(false) == 0); }
TEST(HIPIntegrationTest, PimNRBN3Sync) { EXPECT_TRUE(pim_bn_nr3(true) == 0); }
TEST(HIPIntegrationTest, PimNRBN3Async) { EXPECT_TRUE(pim_bn_nr3(false) == 0); }
TEST(HIPIntegrationTest, PimNRBN4Sync) { EXPECT_TRUE(pim_bn_nr4(true) == 0); }
TEST(HIPIntegrationTest, PimNRBN4Async) { EXPECT_TRUE(pim_bn_nr4(false) == 0); }
TEST(HIPIntegrationTest, PimNRBN5Sync) { EXPECT_TRUE(pim_bn_nr5(true) == 0); }
TEST(HIPIntegrationTest, PimNRBN5Async) { EXPECT_TRUE(pim_bn_nr5(false) == 0); }
TEST(HIPIntegrationTest, PimNRBN6Sync) { EXPECT_TRUE(pim_bn_nr6(true) == 0); }
TEST(HIPIntegrationTest, PimNRBN6Async) { EXPECT_TRUE(pim_bn_nr6(false) == 0); }
TEST(HIPIntegrationTest, PimNRBN7Sync) { EXPECT_TRUE(pim_bn_nr7(true) == 0); }
TEST(HIPIntegrationTest, PimNRBN7Async) { EXPECT_TRUE(pim_bn_nr7(false) == 0); }
TEST(HIPIntegrationTest, PimNRBN8Sync) { EXPECT_TRUE(pim_bn_nr8(true) == 0); }
TEST(HIPIntegrationTest, PimNRBN8Async) { EXPECT_TRUE(pim_bn_nr8(false) == 0); }
#endif
