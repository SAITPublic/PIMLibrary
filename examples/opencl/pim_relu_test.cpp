
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
#include <random>
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

using namespace std;
using half_float::half;

void calculate_relu(half_float::half* input, half_float::half* output, int input_len)
{
    for (int i = 0; i < input_len; i++) {
        output[i] = (input[i] > 0) ? input[0] : 0;
    }
}

bool pim_relu_sync(bool block, uint32_t input_len)
{
    bool ret = true;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_OPENCL, PIM_FP16);

    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, input_len, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    set_rand_half_data((half_float::half*)host_input->data, (half_float::half)0.5, input_len);
    calculate_relu((half_float::half*)host_input->data, (half_float::half*)golden_output->data, input_len);

    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);
    PimExecuteRelu(device_output, pim_input, nullptr, true);
    PimCopyMemory(host_output, device_output, PIM_TO_HOST);
    int compare_result =
        compare_half_relative((half_float::half*)golden_output->data, (half_float::half*)host_output->data, input_len);
    if (compare_result != 0) {
        ret = false;
    }

    PimDestroyBo(host_input);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);
    PimDeinitialize();

    return ret;
}

TEST(OCLPimIntegrationTest, PimRelu1Sync) { EXPECT_TRUE(pim_relu_sync(true, 256 * 1024)); }
TEST(OCLPimIntegrationTest, PimRelu2Sync) { EXPECT_TRUE(pim_relu_sync(true, 1 * 1024)); }
