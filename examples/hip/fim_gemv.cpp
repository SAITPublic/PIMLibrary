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

#define IN_LENGTH (256)
#define OUT_LENGTH (4096)

using namespace std;

int fim_gemv(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(IN_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_reordered_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_input = FimCreateBo(IN_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* device_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, weight, output data */
    load_data("../test_vectors/load/gemv/input_256x1.dat", (char*)host_input->data, host_input->size);
    load_data("../test_vectors/load/gemv/weight_256x4096.dat", (char*)host_weight->data, host_weight->size);
    load_data("../test_vectors/load/gemv/output_4096x1.dat", (char*)host_output->data, host_output->size);
    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    dump_data("../test_vectors/dump/gemv/weight_256x4096.dat", (char*)host_weight->data, host_weight->size);
    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    dump_data("../test_vectors/dump/gemv/preloaded_weight_256x4096.dat", (char*)preloaded_weight->data,
              preloaded_weight->size);

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecute(host_output, device_input, preloaded_weight, OP_GEMV);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}
TEST(IntegrationTest, FimGEMV) { EXPECT_TRUE(fim_gemv() == 0); }
