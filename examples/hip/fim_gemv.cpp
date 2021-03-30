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
#include "utility/fim_dump.hpp"
#include "utility/fim_profile.h"

#define IN_LENGTH (256)
#define OUT_LENGTH (4096)
#define BATCH_DIM (2)

#ifdef DEBUG_FIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using half_float::half;

int fim_gemv_batch(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(IN_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_reordered_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_input = FimCreateBo(IN_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* device_output = FimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* host_output = FimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_HOST);

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

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);

    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (GEMV) */
        FimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);

        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH * BATCH_DIM);
    }

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv_256(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(IN_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_reordered_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_input = FimCreateBo(IN_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* device_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* host_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/input_256x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_256x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_256x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (GEMV) */
        FimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH);
    }
    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv_512(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(IN_LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_weight = FimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_reordered_weight = FimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_input = FimCreateBo(IN_LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* device_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_weight = FimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* host_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);

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

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (GEMV) */
        FimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH);
    }

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv_desc(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimExecuteDummy();

    FimDesc* fim_desc = FimCreateDesc(1, 1, out_size, in_size, FIM_FP16, OP_GEMV);
    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    FimBo* host_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* temp_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* host_reordered_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* device_input = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    FimBo* preloaded_weight = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    FimBo* golden_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/gemv_input_1024x4096.dat";
    std::string weight = test_vector_data + "load/gemv/gemv_weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/gemv_output_1024x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/gemv_preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/gemv_output_1024x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, out_size * sizeof(half));
    for (int i = 0; i < fim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * fim_desc->bshape_r.w, (half*)temp_weight->data + i * fim_desc->bshape.w,
               fim_desc->bshape_r.w * sizeof(half));
    }

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_FIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (GEMV) */
        FimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, out_size);
    }

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(temp_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);
    FimDestroyDesc(fim_desc);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv_desc_batch(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;
    int batch_n = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimDesc* fim_desc = FimCreateDesc(batch_n, 1, out_size, in_size, FIM_FP16, OP_GEMV);
    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    FimBo* host_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* temp_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* host_reordered_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* device_input = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    FimBo* preloaded_weight = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    FimBo* temp_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    FimBo* golden_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/gemv_batch_input_4x1024.dat";
    std::string weight = test_vector_data + "load/gemv/gemv_batch_weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/gemv_batch_output_4x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/gemv_batch_preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/gemv_batch_output_4x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)temp_output->data, temp_output->size);

    for (int i = 0; i < batch_n; i++) {
        memcpy((half*)golden_output->data + i * fim_desc->bshape_r.h, (half*)temp_output->data + i * fim_desc->bshape.h,
               fim_desc->bshape_r.h * sizeof(half));
    }

    for (int i = 0; i < fim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * fim_desc->bshape_r.w, (half*)temp_weight->data + i * fim_desc->bshape.w,
               fim_desc->bshape_r.w * sizeof(half));
    }

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_FIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (GEMV) */
        FimExecuteGemv(device_output, device_input, preloaded_weight, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, OUT_LENGTH * BATCH_DIM);
    }
    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(temp_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(temp_output);
    FimDestroyBo(golden_output);
    FimDestroyDesc(fim_desc);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv_lut(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimExecuteDummy();

    FimDesc* fim_desc = FimCreateDesc(1, 1, out_size, in_size, FIM_FP16, OP_GEMV);
    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    FimBo* host_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* temp_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* host_reordered_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* device_input = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    FimBo* preloaded_weight = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    FimBo* golden_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    FimBo* dev_in;
    FimBo* dev_out;
    FimBo* fim_weight;

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/gemv_input_1024x4096.dat";
    std::string weight = test_vector_data + "load/gemv/gemv_weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/gemv_output_1024x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/gemv_preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/gemv_output_1024x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, out_size * sizeof(half));

    for (int i = 0; i < fim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * fim_desc->bshape_r.w, (half*)temp_weight->data + i * fim_desc->bshape.w,
               fim_desc->bshape_r.w * sizeof(half));
    }

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_FIM);

    FimInsertGemvBundle(reinterpret_cast<uint64_t>(host_weight->data),
                        FimCreateGemvBundle(device_input, preloaded_weight, device_output));

    FimGemvBundle* bundle = FimFindGemvBundle(reinterpret_cast<uint64_t>(host_weight->data));
    dev_in = bundle->in;
    dev_out = bundle->out;
    fim_weight = bundle->wei;

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGemv(dev_out, dev_in, fim_weight, nullptr, block);
    if (!block) FimSynchronize();
    FimCopyMemory(host_output, dev_out, DEVICE_TO_HOST);

    //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, out_size);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(temp_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);
    FimDestroyDesc(fim_desc);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();  // device_input, preloaded_weight, device_output will be destroyed in FimDeinitialize()

    return ret;
}

int fim_gemv_lut_profile(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimExecuteDummy();

    FimDesc* fim_desc = FimCreateDesc(1, 1, out_size, in_size, FIM_FP16, OP_GEMV);
    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    FimBo* host_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* temp_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* host_reordered_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* device_input = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    FimBo* preloaded_weight = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    FimBo* golden_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    FimBo* dev_in;
    FimBo* dev_out;
    FimBo* fim_weight;

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/gemv_input_1024x4096.dat";
    std::string weight = test_vector_data + "load/gemv/gemv_weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/gemv_output_1024x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/gemv_preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/gemv_output_1024x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, out_size * sizeof(half));

    for (int i = 0; i < fim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * fim_desc->bshape_r.w, (half*)temp_weight->data + i * fim_desc->bshape.w,
               fim_desc->bshape_r.w * sizeof(half));
    }
    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_FIM);
    FimInsertGemvBundle(reinterpret_cast<uint64_t>(host_weight->data),
                        FimCreateGemvBundle(device_input, preloaded_weight, device_output));

    int iter;
#ifdef TARGET
    FIM_PROFILE_TICK_A(GEMV_E2E);
    for (iter = 0; iter < 1000; iter++) {
#else
    for (iter = 0; iter < 1; iter++) {
#endif
        FimGemvBundle* bundle = FimFindGemvBundle(reinterpret_cast<uint64_t>(host_weight->data));
        dev_in = bundle->in;
        dev_out = bundle->out;
        fim_weight = bundle->wei;

        /* __FIM_API__ call : Execute FIM kernel (GEMV) */
        FimExecuteGemv(dev_out, dev_in, fim_weight);
    }
    FimSynchronize();
#ifdef TARGET
    FIM_PROFILE_TOCK_A(GEMV_E2E);
    printf("[ %d execution time ]\n", iter);
#endif

    FimCopyMemory(host_output, dev_out, DEVICE_TO_HOST);
    //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, out_size);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(temp_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);
    FimDestroyDesc(fim_desc);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();  // device_input, preloaded_weight, device_output will be destroyed in FimDeinitialize()

    return ret;
}

TEST(HIPIntegrationTest, FimGemvBatchSync) { EXPECT_TRUE(fim_gemv_batch(true) == 0); }
TEST(HIPIntegrationTest, FimGemvBatchAsync) { EXPECT_TRUE(fim_gemv_batch(false) == 0); }
TEST(HIPIntegrationTest, FimGemv256Sync) { EXPECT_TRUE(fim_gemv_256(true) == 0); }
TEST(HIPIntegrationTest, FimGemv256Async) { EXPECT_TRUE(fim_gemv_256(false) == 0); }
TEST(HIPIntegrationTest, FimGemv512Sync) { EXPECT_TRUE(fim_gemv_512(true) == 0); }
TEST(HIPIntegrationTest, FimGemv512Async) { EXPECT_TRUE(fim_gemv_512(false) == 0); }
TEST(HIPIntegrationTest, FimGemvDescSync) { EXPECT_TRUE(fim_gemv_desc(true) == 0); }
TEST(HIPIntegrationTest, FimGemvDescAsync) { EXPECT_TRUE(fim_gemv_desc(false) == 0); }
TEST(HIPIntegrationTest, FimGemvDescBatchSync) { EXPECT_TRUE(fim_gemv_desc_batch(true) == 0); }
TEST(HIPIntegrationTest, FimGemvDescBatchASync) { EXPECT_TRUE(fim_gemv_desc_batch(false) == 0); }
TEST(HIPIntegrationTest, FimGemvLutSync) { EXPECT_TRUE(fim_gemv_lut(true) == 0); }
TEST(HIPIntegrationTest, FimGemvLutAsync) { EXPECT_TRUE(fim_gemv_lut(false) == 0); }
TEST(HIPIntegrationTest, FimGemvLutProfileAsync) { EXPECT_TRUE(fim_gemv_lut_profile(false) == 0); }
