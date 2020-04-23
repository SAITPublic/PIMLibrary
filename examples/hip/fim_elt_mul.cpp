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

#define LENGTH (64 * 1024)

using namespace std;

int fim_elt_mul_1(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_fim_input = FimCreateBo(2 * LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, weight, output data */
    load_data("../test_vectors/load/elt_mul/input0_128KB.dat", (char*)host_input0->data, host_input0->size);
    load_data("../test_vectors/load/elt_mul/input1_128KB.dat", (char*)host_input1->data, host_input1->size);
    load_data("../test_vectors/load/elt_mul/output_128KB.dat", (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(preloaded_fim_input, host_input0, host_input1, OP_ELT_MUL);

    /* __FIM_API__ call : Execute FIM kernel (ELT_MUL) */
    FimExecuteMul(device_output, preloaded_fim_input);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    dump_data("../test_vectors/dump/elt_mul/preloaded_input_256KB.dat", (char*)preloaded_fim_input->data,
              preloaded_fim_input->size);
    dump_data("../test_vectors/dump/elt_mul/output_128KB.dat", (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input0);
    FimDestroyBo(host_input1);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_elt_mul_2(void)
{
    int ret = 0;

    FimBo host_input0 = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo host_input1 = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo host_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo golden_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo device_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_DEVICE};
    FimBo preloaded_fim_input = {.size = 2 * LENGTH * sizeof(half), .mem_type = MEM_TYPE_FIM};

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate memory */
    FimAllocMemory(&host_input0);
    FimAllocMemory(&host_input1);
    FimAllocMemory(&host_output);
    FimAllocMemory(&golden_output);
    FimAllocMemory(&device_output);
    FimAllocMemory(&preloaded_fim_input);

    /* Initialize the input, weight, output data */
    load_data("../test_vectors/load/elt_mul/input0_128KB.dat", (char*)host_input0.data, host_input0.size);
    load_data("../test_vectors/load/elt_mul/input1_128KB.dat", (char*)host_input1.data, host_input1.size);
    load_data("../test_vectors/load/elt_mul/output_128KB.dat", (char*)golden_output.data, golden_output.size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(&preloaded_fim_input, &host_input0, &host_input1, OP_ELT_MUL);

    /* __FIM_API__ call : Execute FIM kernel (ELT_MUL) */
    FimExecuteMul(&device_output, &preloaded_fim_input);

    FimCopyMemory(&host_output, &device_output, DEVICE_TO_HOST);

    ret = compare_data((char*)golden_output.data, (char*)host_output.data, host_output.size);

    dump_data("../test_vectors/dump/elt_mul/preloaded_input_256KB.dat", (char*)preloaded_fim_input.data,
              preloaded_fim_input.size);
    dump_data("../test_vectors/dump/elt_mul/output_128KB.dat", (char*)host_output.data, host_output.size);

    /* __FIM_API__ call : Free memory */
    FimFreeMemory(&host_input0);
    FimFreeMemory(&host_input1);
    FimFreeMemory(&host_output);
    FimFreeMemory(&golden_output);
    FimFreeMemory(&device_output);
    FimFreeMemory(&preloaded_fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_elt_mul_3(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_fim_input = FimCreateBo(2 * LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, weight, output data */
    load_data("../test_vectors/load/elt_mul/mul_input0_256KB.dat", (char*)host_input0->data, host_input0->size);
    load_data("../test_vectors/load/elt_mul/mul_input1_256KB.dat", (char*)host_input1->data, host_input1->size);
    load_data("../test_vectors/load/elt_mul/mul_output_256KB.dat", (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(preloaded_fim_input, host_input0, host_input1, OP_ELT_MUL);

    /* __FIM_API__ call : Execute FIM kernel (ELT_MUL) */
    FimExecuteMul(device_output, preloaded_fim_input);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    dump_data("../test_vectors/dump/elt_mul/preloaded_input_512KB.dat", (char*)preloaded_fim_input->data,
              preloaded_fim_input->size);
    dump_data("../test_vectors/dump/elt_mul/output_256KB.dat", (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input0);
    FimDestroyBo(host_input1);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, FimEltMul1) { EXPECT_TRUE(fim_elt_mul_1() == 0); }
TEST(HIPIntegrationTest, FimEltMul2) { EXPECT_TRUE(fim_elt_mul_2() == 0); }
TEST(HIPIntegrationTest, FimEltMul3) { EXPECT_TRUE(fim_elt_mul_3() == 0); }
