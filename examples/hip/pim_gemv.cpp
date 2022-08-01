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

#define IN_LENGTH (256)
#define OUT_LENGTH (4096)
#define BATCH_DIM (2)

#ifdef DEBUG_PIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using half_float::half;

#define EPSILON (1.0)

int pim_gemv_batch(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(BATCH_DIM, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(BATCH_DIM, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(BATCH_DIM, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(BATCH_DIM, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(BATCH_DIM, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/batch_input_2x256.dat";
    std::string weight = test_vector_data + "load/gemv/batch_weight_256x4096.dat";
    std::string output = test_vector_data + "load/gemv/batch_output_2x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/batch_preloaded_weight_256x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/batch_output_2x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);

        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH * BATCH_DIM,
                                    EPSILON);
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_256(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(1, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/input_256x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_256x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_256x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1_256.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
        // dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH, EPSILON);
    }
    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_512(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(1, 1, 1, IN_LENGTH * 2, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, IN_LENGTH * 2, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, 1, 1, IN_LENGTH * 2, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, IN_LENGTH * 2, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/input_512x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_512x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1_512.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_512x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1_512.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH, EPSILON);
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_desc(bool block)
{
    int ret = 0;
    int in_size = 1024;
    int out_size = 4096;
    float alpha = 1.0f;
    float beta = 0.0f;
    //float epsilon = 0.1f; // not-used

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimDesc* pim_desc = PimCreateDesc(1, 1, in_size, out_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    set_half_data((half*)golden_output->data, half(0.0), out_size);
    set_half_data((half*)host_output->data, half(0.0), out_size);
    set_half_data((half*)host_input->data, half(dis(gen)), in_size);
    set_half_data((half*)host_weight->data, half(dis(gen)), in_size * out_size);
    matmulCPU((half*)host_input->data, (half*)host_weight->data, (half*)golden_output->data, 1, out_size, in_size,
              half(alpha), half(beta));

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMV) */
    ret = PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
    if (!block) PimSynchronize();

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, EPSILON);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_desc_batch(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;
    int batch_n = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimDesc* pim_desc = PimCreateDesc(batch_n, 1, in_size, out_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* temp_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/batch_input_4x1024.dat";
    std::string weight = test_vector_data + "load/gemv/batch_weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/batch_output_4x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/gemv_batch_preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/gemv_batch_output_4x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)temp_output->data, temp_output->size);

    for (int i = 0; i < batch_n; i++) {
        memcpy((half*)golden_output->data + i * pim_desc->bshape_r.w, (half*)temp_output->data + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }

    for (int i = 0; i < pim_desc->bshape_r.w; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.h, (half*)temp_weight->data + i * pim_desc->bshape.h,
               pim_desc->bshape_r.h * sizeof(half));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size * batch_n, EPSILON);
    }
    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(temp_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_uniform_128(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(1, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    // input is 0 from 128 to 255
    std::string input = test_vector_data + "load/gemv/uniform_input_256x1.dat";
    std::string weight = test_vector_data + "load/gemv/uniform_weight_256x4096.dat";
    std::string output = test_vector_data + "load/gemv/uniform_output_4096x1.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/uniform_preloaded_weight_256x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/uniform_output_4096x1.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH, EPSILON);
    }
    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_normal_128(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(1, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    // input is 0 from 128 to 255
    std::string input = test_vector_data + "load/gemv/normal_input_256x1.dat";
    std::string weight = test_vector_data + "load/gemv/normal_weight_256x4096.dat";
    std::string output = test_vector_data + "load/gemv/normal_output_4096x1.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/normal_preloaded_weight_256x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/normal_output_4096x1.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH, EPSILON);
    }
    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_uniform_4096(bool block)
{
    int ret = 0;
    int input_length = 4096;
    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(1, 1, 1, input_length, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, input_length, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, 1, 1, input_length, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, input_length, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/uniform_input_4096x1.dat";
    std::string weight = test_vector_data + "load/gemv/uniform_weight_4096x4096.dat";
    std::string output = test_vector_data + "load/gemv/uniform_output_4096x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/uniform_preloaded_weight_4096x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/uniform_output_4096x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH, EPSILON);
    }
    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_normal_4096(bool block)
{
    int ret = 0;
    int input_length = 4096;
    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(1, 1, 1, input_length, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, input_length, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, 1, 1, input_length, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, input_length, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    // input is 0 from 128 to 255
    std::string input = test_vector_data + "load/gemv/normal_input_4096x1.dat";
    std::string weight = test_vector_data + "load/gemv/normal_weight_4096x4096.dat";
    std::string output = test_vector_data + "load/gemv/normal_output_4096x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/normal_preloaded_weight_4096x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/normal_output_4096x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH, EPSILON);
    }
    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_no_accum_512(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(1, 1, 1, IN_LENGTH * 2, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, IN_LENGTH * 2, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, 1, 1, IN_LENGTH * 2, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, IN_LENGTH * 2, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/input_512x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_512x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1_512.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_512x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1_512.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH, EPSILON);
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_no_accum_256(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(1, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, 1, 1, IN_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, 1, IN_LENGTH, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, 1, 1, OUT_LENGTH, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/input_256x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_256x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_256x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1_256.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH, EPSILON);
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_no_accum_desc(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimDesc* pim_desc = PimCreateDesc(1, 1, in_size, out_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/input_1024x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1_1024.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1_1024.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, out_size * sizeof(half));
    for (int i = 0; i < pim_desc->bshape_r.w; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.h, (half*)temp_weight->data + i * pim_desc->bshape.h,
               pim_desc->bshape_r.h * sizeof(half));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, EPSILON);
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */

    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_moe(bool block)
{
    /* This test case is to accelerate Mixture of Expert */
    int ret = 0;
    int in_size = 256;
    int out_size = 512;
    int moe_cnt = 8;
    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    /* __PIM_API__ call : Create PIM Buffer Object List */
    PimBo* host_input = PimCreateBo(1, moe_cnt, 1, in_size, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, moe_cnt, in_size, out_size, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, moe_cnt, 1, out_size, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, moe_cnt, 1, out_size, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, moe_cnt, 1, in_size, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, moe_cnt, in_size, out_size, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, moe_cnt, 1, out_size, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    set_half_data((half*)golden_output->data, half(0.0), out_size * moe_cnt);
    set_half_data((half*)host_output->data, half(0.0), out_size * moe_cnt);
    for (int i = 0; i < moe_cnt; i++) {
        set_half_data(((half*)host_input->data) + i * in_size, half(dis(gen)), in_size);
        set_half_data(((half*)host_weight->data) + i * in_size * out_size, half(dis(gen)), in_size * out_size);
        matmulCPU(((half*)host_input->data) + i * in_size, ((half*)host_weight->data) + i * in_size * out_size,
                  ((half*)golden_output->data) + i * out_size, 1, out_size, in_size, half(alpha), half(beta));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMV list) */
    PimExecuteGemvList(device_output, device_input, device_weight, nullptr, block);
    if (!block) PimSynchronize();

    /* check output result */
    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    for (int i = 0; i < moe_cnt; i++) {
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size * moe_cnt, epsilon);
        if (ret != 0) break;
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_moe_chwise(bool block)
{
    /* This test case is to accelerate Mixture of Expert */
    int ret = 0;
    int in_size = 256;
    int out_size = 64;
    int moe_cnt = 64;
    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 0.1f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    /* __PIM_API__ call : Create PIM Buffer Object List */
    PimBo* host_input = PimCreateBo(1, moe_cnt, 1, in_size, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(1, moe_cnt, in_size, out_size, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(1, moe_cnt, 1, out_size, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(1, moe_cnt, 1, out_size, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(1, moe_cnt, 1, in_size, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_weight = PimCreateBo(1, moe_cnt, in_size, out_size, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(1, moe_cnt, 1, out_size, PIM_FP16, MEM_TYPE_DEVICE);

    /* Initialize the input, weight, output data */
    set_half_data((half*)golden_output->data, half(0.0), out_size * moe_cnt);
    set_half_data((half*)host_output->data, half(0.0), out_size * moe_cnt);
    for (int i = 0; i < moe_cnt; i++) {
        set_half_data(((half*)host_input->data) + i * in_size, half(dis(gen)), in_size);
        set_half_data(((half*)host_weight->data) + i * in_size * out_size, half(dis(gen)), in_size * out_size);
        matmulCPU(((half*)host_input->data) + i * in_size, ((half*)host_weight->data) + i * in_size * out_size,
                  ((half*)golden_output->data) + i * out_size, 1, out_size, in_size, half(alpha), half(beta));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMV list) */
    PimExecuteGemvList(device_output, device_input, device_weight, nullptr, block);
    if (!block) PimSynchronize();

    /* check output result */
    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size * moe_cnt, epsilon);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, PimGemvMoEChwiseSync) { EXPECT_TRUE(pim_gemv_moe_chwise(true) == 0); }
TEST(HIPIntegrationTest, PimGemvMoESync) { EXPECT_TRUE(pim_gemv_moe(true) == 0); }
TEST(HIPIntegrationTest, PimGemvBatchSync) { EXPECT_TRUE(pim_gemv_batch(true) == 0); }
TEST(HIPIntegrationTest, PimGemvBatchAsync) { EXPECT_TRUE(pim_gemv_batch(false) == 0); }
TEST(HIPIntegrationTest, PimGemv256Sync) { EXPECT_TRUE(pim_gemv_256(true) == 0); }
TEST(HIPIntegrationTest, PimGemv256Async) { EXPECT_TRUE(pim_gemv_256(false) == 0); }
TEST(HIPIntegrationTest, PimGemv512Sync) { EXPECT_TRUE(pim_gemv_512(true) == 0); }
TEST(HIPIntegrationTest, PimGemv512Async) { EXPECT_TRUE(pim_gemv_512(false) == 0); }
TEST(HIPIntegrationTest, PimGemvDescSync) { EXPECT_TRUE(pim_gemv_desc(true) == 0); }
TEST(HIPIntegrationTest, PimGemvDescAsync) { EXPECT_TRUE(pim_gemv_desc(false) == 0); }
TEST(HIPIntegrationTest, PimGemvDescBatchSync) { EXPECT_TRUE(pim_gemv_desc_batch(true) == 0); }
TEST(HIPIntegrationTest, PimGemvDescBatchASync) { EXPECT_TRUE(pim_gemv_desc_batch(false) == 0); }
TEST(HIPIntegrationTest, PimGemvUniform128Sync) { EXPECT_TRUE(pim_gemv_uniform_128(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNormal128Sync) { EXPECT_TRUE(pim_gemv_normal_128(true) == 0); }
TEST(HIPIntegrationTest, PimGemvUniform4096Sync) { EXPECT_TRUE(pim_gemv_uniform_4096(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNormal4096Sync) { EXPECT_TRUE(pim_gemv_normal_4096(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNoAccum512Sync) { EXPECT_TRUE(pim_gemv_no_accum_512(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNoAccum256Sync) { EXPECT_TRUE(pim_gemv_no_accum_256(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNoAccumDescSync) { EXPECT_TRUE(pim_gemv_no_accum_desc(true) == 0); }
