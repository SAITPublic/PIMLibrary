#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "half.hpp"
#include "utility/fim_dump.hpp"

#define LENGTH (128 * 1024)

#ifdef DEBUG_FIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

int fim_relu_1(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/relu/input_256KB.dat";
    std::string output = test_vector_data + "load/relu/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/relu/output_256KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel */
        FimExecuteRelu(device_output, fim_input, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, FIM_TO_HOST);

        //    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);
        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data,
                                     host_output->size / sizeof(half));
    }
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, FimRelu1Sync) { EXPECT_TRUE(fim_relu_1(true) == 0); }
TEST(HIPIntegrationTest, FimRelu1Async) { EXPECT_TRUE(fim_relu_1(false) == 0); }
