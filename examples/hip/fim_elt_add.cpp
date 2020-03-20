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

int fim_elt_add(void)
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
    load_data("../test_vectors/load/elt_add/input0_128KB.dat", (char*)host_input0.data, host_input0.size);
    load_data("../test_vectors/load/elt_add/input1_128KB.dat", (char*)host_input1.data, host_input1.size);
    load_data("../test_vectors/load/elt_add/output_128KB.dat", (char*)golden_output.data, golden_output.size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(&preloaded_fim_input, &host_input0, &host_input1, OP_ELT_ADD);

    /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
    FimExecute(&device_output, &preloaded_fim_input, OP_ELT_ADD);

    FimCopyMemory(&host_output, &device_output, DEVICE_TO_HOST);

    ret = compare_data((char*)golden_output.data, (char*)host_output.data, host_output.size);

    dump_data("../test_vectors/dump/elt_add/preloaded_input_256KB.dat", (char*)preloaded_fim_input.data,
              preloaded_fim_input.size);
    dump_data("../test_vectors/dump/elt_add/output_128KB.dat", (char*)host_output.data, host_output.size);

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
TEST(IntegrationTest, FimEltAdd) { EXPECT_TRUE(fim_elt_add() == 0); }
