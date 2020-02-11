#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"
#include "utility/fim_dump.hpp"

#define LENGTH (64 * 1024)

using namespace std;

int verifyResult(FimBo& hostInput, FimBo& hostWeight, FimBo& hostOutput, FimBo& deviceOutput);

int fim_elt_add(void)
{
    int ret = 0;

    FimBo hostInput = {.size = LENGTH * sizeof(FP16), .memType = MEM_TYPE_HOST};
    FimBo hostWeight = {.size = LENGTH * sizeof(FP16), .memType = MEM_TYPE_HOST};
    FimBo hostOutput = {.size = LENGTH * sizeof(FP16), .memType = MEM_TYPE_HOST};
    FimBo deviceOutput = {.size = LENGTH * sizeof(FP16), .memType = MEM_TYPE_DEVICE};
    FimBo fimWeight = {.size = 2 * LENGTH * sizeof(FP16), .memType = MEM_TYPE_FIM};

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate host(CPU) memory */
    FimAllocMemory(&hostInput);
    FimAllocMemory(&hostWeight);
    FimAllocMemory(&hostOutput);
    /* __FIM_API__ call : Allocate device(GPU) memory */
    FimAllocMemory(&deviceOutput);
    /* __FIM_API__ call : Allocate device(FIM) memory */
    FimAllocMemory(&fimWeight);

    /* Initialize the input, weight, output data */
    load_fp16_data("../test_vectors/load/elt_add_input0_64K_fp16.txt", (FP16*)hostInput.data, hostInput.size);
    load_fp16_data("../test_vectors/load/elt_add_input1_64K_fp16.txt", (FP16*)hostWeight.data, hostWeight.size);
    load_fp16_data("../test_vectors/load/elt_add_output_64K_fp16.txt", (FP16*)hostOutput.data, hostOutput.size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(&fimWeight, &hostInput, &hostWeight, OP_ELT_ADD);

    dump_fp16_data("../test_vectors/dump/elt_add_preloaded_input_128K_fp16.txt", (FP16*)fimWeight.data, fimWeight.size);

    /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
    FimExecute(&deviceOutput, &fimWeight, OP_ELT_ADD);

    /* __FIM_API__ call : Free device(FIM) memory */
    FimFreeMemory(&fimWeight);
    /* __FIM_API__ call : Free host(CPU) memory */
    FimFreeMemory(&hostInput);
    FimFreeMemory(&hostWeight);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}
TEST(IntegrationTest, FimEltAdd) { EXPECT_TRUE(fim_elt_add() == 0); }
