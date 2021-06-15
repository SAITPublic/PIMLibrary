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
#include "pim_runtime_api.h"
#include "utility/pim_dump.hpp"
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

#define EPSILON (0.05)

int pim_gemv_batch(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(IN_LENGTH, 1, 1, BATCH_DIM, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(IN_LENGTH, 1, 1, BATCH_DIM, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);

    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);

        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH * BATCH_DIM,
                                    EPSILON);
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

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
    PimBo* host_input = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

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
    PimBo* host_input = PimCreateBo(IN_LENGTH * 2, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(IN_LENGTH * 2, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_desc(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimDesc* pim_desc = PimCreateDesc(1, 1, out_size, in_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_reordered_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    PimBo* preloaded_weight = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

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
    for (int i = 0; i < pim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.w, (half*)temp_weight->data + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_PIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);
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

    PimDesc* pim_desc = PimCreateDesc(batch_n, 1, out_size, in_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_reordered_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    PimBo* preloaded_weight = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* temp_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

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
        memcpy((half*)golden_output->data + i * pim_desc->bshape_r.h, (half*)temp_output->data + i * pim_desc->bshape.h,
               pim_desc->bshape_r.h * sizeof(half));
    }

    for (int i = 0; i < pim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.w, (half*)temp_weight->data + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_PIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH * BATCH_DIM,
                                    EPSILON);
    }
    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(temp_output);
    PimDestroyBo(golden_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_lut(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimDesc* pim_desc = PimCreateDesc(1, 1, out_size, in_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_reordered_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    PimBo* preloaded_weight = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    PimBo* dev_in;
    PimBo* dev_out;
    PimBo* pim_weight;

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

    for (int i = 0; i < pim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.w, (half*)temp_weight->data + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_PIM);

    PimInsertGemvBundle(reinterpret_cast<uint64_t>(host_weight->data),
                        PimCreateGemvBundle(device_input, preloaded_weight, device_output));

    PimGemvBundle* bundle = PimFindGemvBundle(reinterpret_cast<uint64_t>(host_weight->data));
    dev_in = bundle->in;
    dev_out = bundle->out;
    pim_weight = bundle->wei;

    /* __PIM_API__ call : Execute PIM kernel (GEMV) */
    PimExecuteGemv(dev_out, dev_in, pim_weight, nullptr, block);
    if (!block) PimSynchronize();
    PimCopyMemory(host_output, dev_out, DEVICE_TO_HOST);

    //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, EPSILON);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();  // device_input, preloaded_weight, device_output will be destroyed in PimDeinitialize()

    return ret;
}

int pim_gemv_lut_profile(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimDesc* pim_desc = PimCreateDesc(1, 1, out_size, in_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_reordered_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    PimBo* preloaded_weight = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    PimBo* dev_in;
    PimBo* dev_out;
    PimBo* pim_weight;

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

    for (int i = 0; i < pim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.w, (half*)temp_weight->data + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }
    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_PIM);
    PimInsertGemvBundle(reinterpret_cast<uint64_t>(host_weight->data),
                        PimCreateGemvBundle(device_input, preloaded_weight, device_output));

    int iter;
#ifdef TARGET
    PIM_PROFILE_TICK_A(GEMV_E2E);
    for (iter = 0; iter < 1000; iter++) {
#else
    for (iter = 0; iter < 1; iter++) {
#endif
        PimGemvBundle* bundle = PimFindGemvBundle(reinterpret_cast<uint64_t>(host_weight->data));
        dev_in = bundle->in;
        dev_out = bundle->out;
        pim_weight = bundle->wei;

        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(dev_out, dev_in, pim_weight);
    }
    PimSynchronize();
#ifdef TARGET
    PIM_PROFILE_TOCK_A(GEMV_E2E);
    printf("[ %d execution time ]\n", iter);
#endif

    PimCopyMemory(host_output, dev_out, DEVICE_TO_HOST);
    //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, EPSILON);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();  // device_input, preloaded_weight, device_output will be destroyed in PimDeinitialize()

    return ret;
}

int pim_gemv_uniform_128(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

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
    PimBo* host_input = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

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
    PimBo* host_input = PimCreateBo(input_length, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(input_length, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(input_length, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(input_length, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(input_length, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

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
    PimBo* host_input = PimCreateBo(input_length, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(input_length, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(input_length, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(input_length, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(input_length, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

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
    PimBo* host_input = PimCreateBo(IN_LENGTH * 2, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(IN_LENGTH * 2, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

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
    PimBo* host_input = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_reordered_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_input = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* device_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* preloaded_weight = PimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(OUT_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

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

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
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
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

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

    PimDesc* pim_desc = PimCreateDesc(1, 1, out_size, in_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_reordered_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    PimBo* preloaded_weight = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

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
    for (int i = 0; i < pim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.w, (half*)temp_weight->data + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    PimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_PIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, EPSILON);
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */

    PimDestroyDesc(pim_desc);
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);
    PimDestroyBo(preloaded_weight);
    PimDestroyBo(host_reordered_weight);
    PimDestroyBo(golden_output);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

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
TEST(HIPIntegrationTest, PimGemvLutSync) { EXPECT_TRUE(pim_gemv_lut(true) == 0); }
TEST(HIPIntegrationTest, PimGemvLutAsync) { EXPECT_TRUE(pim_gemv_lut(false) == 0); }
TEST(HIPIntegrationTest, PimGemvLutProfileAsync) { EXPECT_TRUE(pim_gemv_lut_profile(false) == 0); }
TEST(HIPIntegrationTest, PimGemvUniform128Sync) { EXPECT_TRUE(pim_gemv_uniform_128(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNormal128Sync) { EXPECT_TRUE(pim_gemv_normal_128(true) == 0); }
TEST(HIPIntegrationTest, PimGemvUniform4096Sync) { EXPECT_TRUE(pim_gemv_uniform_4096(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNormal4096Sync) { EXPECT_TRUE(pim_gemv_normal_4096(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNoAccum512Sync) { EXPECT_TRUE(pim_gemv_no_accum_512(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNoAccum256Sync) { EXPECT_TRUE(pim_gemv_no_accum_256(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNoAccumDescSync) { EXPECT_TRUE(pim_gemv_no_accum_desc(true) == 0); }