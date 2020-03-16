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
    FimBo device_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_DEVICE};
    FimBo preloaded_fim_input = {.size = 2 * LENGTH * sizeof(half), .mem_type = MEM_TYPE_FIM};

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate host(CPU) memory */
    FimAllocMemory(&host_input0);
    FimAllocMemory(&host_input1);
    FimAllocMemory(&host_output);
    /* __FIM_API__ call : Allocate device(GPU) memory */
    FimAllocMemory(&device_output);
    /* __FIM_API__ call : Allocate device(FIM) memory */
    FimAllocMemory(&preloaded_fim_input);

    /* Initialize the input, weight, output data */
    load_data("../test_vectors/load/elt_add_input0_64KB.txt", (char*)host_input0.data, host_input0.size);
    load_data("../test_vectors/load/elt_add_input1_64KB.txt", (char*)host_input1.data, host_input1.size);
    load_data("../test_vectors/load/elt_add_output_64KB.txt", (char*)host_output.data, host_output.size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(&preloaded_fim_input, &host_input0, &host_input1, OP_ELT_ADD);

    dump_data("../test_vectors/dump/elt_add_preloaded_input_256KB.txt", (char*)preloaded_fim_input.data, preloaded_fim_input.size);

    /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
    FimExecute(&device_output, &preloaded_fim_input, OP_ELT_ADD);

    /* __FIM_API__ call : Free device(FIM) memory */
    FimFreeMemory(&preloaded_fim_input);
    /* __FIM_API__ call : Free host(CPU) memory */
    FimFreeMemory(&host_input0);
    FimFreeMemory(&host_input1);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}
TEST(IntegrationTest, FimEltAdd) { EXPECT_TRUE(fim_elt_add() == 0); }
