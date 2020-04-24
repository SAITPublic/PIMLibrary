#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "half.hpp"
#include "utility/fim_dump.hpp"

using half_float::half;

#define LENGTH (128 * 1024)

using namespace std;

int fim_relu_1(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_fim_input = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    load_data("../test_vectors/load/relu/input_256KB.dat", (char*)host_input->data, host_input->size);
    load_data("../test_vectors/load/relu/output_256KB.dat", (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(preloaded_fim_input, host_input, OP_RELU);

    /* __FIM_API__ call : Execute FIM kernel */
    FimExecuteRelu(device_output, preloaded_fim_input);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    dump_data("../test_vectors/dump/relu/preloaded_input_256KB.dat", (char*)preloaded_fim_input->data,
              preloaded_fim_input->size);
    dump_data("../test_vectors/dump/relu/output_256KB.dat", (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_relu_2(void)
{
    int ret = 0;

    FimBo host_input = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo host_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo golden_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo device_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_DEVICE};
    FimBo preloaded_fim_input = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_FIM};

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate memory */
    FimAllocMemory(&host_input);
    FimAllocMemory(&host_output);
    FimAllocMemory(&golden_output);
    FimAllocMemory(&device_output);
    FimAllocMemory(&preloaded_fim_input);

    /* Initialize the input, weight, output data */
    load_data("../test_vectors/load/relu/input_256KB.dat", (char*)host_input.data, host_input.size);
    load_data("../test_vectors/load/relu/output_256KB.dat", (char*)golden_output.data, golden_output.size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(&preloaded_fim_input, &host_input, OP_RELU);

    /* __FIM_API__ call : Execute FIM kernel */
    FimExecuteRelu(&device_output, &preloaded_fim_input);

    FimCopyMemory(&host_output, &device_output, DEVICE_TO_HOST);

    ret = compare_data((char*)golden_output.data, (char*)host_output.data, host_output.size);

    dump_data("../test_vectors/dump/relu/preloaded_input_256KB.dat", (char*)preloaded_fim_input.data,
              preloaded_fim_input.size);
    dump_data("../test_vectors/dump/relu/output_256KB.dat", (char*)host_output.data, host_output.size);

    /* __FIM_API__ call : Free memory */
    FimFreeMemory(&host_input);
    FimFreeMemory(&host_output);
    FimFreeMemory(&golden_output);
    FimFreeMemory(&device_output);
    FimFreeMemory(&preloaded_fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_relu_3(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_fim_input = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    load_data("../test_vectors/load/relu/input_512KB.dat", (char*)host_input->data, host_input->size);
    load_data("../test_vectors/load/relu/output_512KB.dat", (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(preloaded_fim_input, host_input, OP_RELU);

    /* __FIM_API__ call : Execute FIM kernel */
    FimExecuteRelu(device_output, preloaded_fim_input);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    dump_data("../test_vectors/dump/relu/preloaded_input_512KB.dat", (char*)preloaded_fim_input->data,
              preloaded_fim_input->size);
    dump_data("../test_vectors/dump/relu/output_512KB.dat", (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

TEST(IntegrationTest, FimRelu1) { EXPECT_TRUE(fim_relu_1() == 0); }
TEST(IntegrationTest, FimRelu2) { EXPECT_TRUE(fim_relu_2() == 0); }
TEST(IntegrationTest, FimRelu3) { EXPECT_TRUE(fim_relu_3() == 0); }
