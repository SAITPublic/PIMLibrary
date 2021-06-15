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

#define LENGTH (128 * 1024)

#ifdef DEBUG_PIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

int pim_relu_1(bool block)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/relu/input_256KB.dat";
    std::string output = test_vector_data + "load/relu/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/relu/output_256KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel */
        PimExecuteRelu(device_output, pim_input, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, PIM_TO_HOST);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data,
                                    host_output->size / sizeof(half));
    }
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, PimRelu1Sync) { EXPECT_TRUE(pim_relu_1(true) == 0); }
TEST(HIPIntegrationTest, PimRelu1Async) { EXPECT_TRUE(pim_relu_1(false) == 0); }
