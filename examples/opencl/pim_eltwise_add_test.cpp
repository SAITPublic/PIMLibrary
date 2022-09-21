
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
// #include "half.hpp"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

#define IN_LENGTH 256 * 1024

using namespace std;

bool pim_eltwise_add_sync()
{
    bool ret = true;

    PimInitialize(RT_TYPE_OPENCL, PIM_FP16);

    PimBo* host_opr1 = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_opr2 = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_out = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* ref_out = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_opr1 = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* device_opr2 = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(IN_LENGTH, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);

    set_rand_half_data((half_float::half*)host_opr1->data, (half_float::half)0.5, IN_LENGTH);
    set_rand_half_data((half_float::half*)host_opr2->data, (half_float::half)0.5, IN_LENGTH);

    addCPU((half_float::half*)host_opr1->data, (half_float::half*)host_opr2->data, (half_float::half*)ref_out->data,
           IN_LENGTH);
    PimCopyMemory(device_opr1, host_opr1, HOST_TO_PIM);
    PimCopyMemory(device_opr2, host_opr2, HOST_TO_PIM);

    PimExecuteAdd(device_output, device_opr1, device_opr2, nullptr, true);
    PimCopyMemory(host_out, device_output, PIM_TO_HOST);
    int compare_res =
        compare_half_relative((half_float::half*)host_out->data, (half_float::half*)ref_out->data, IN_LENGTH);
    if (compare_res != 0) {
        ret = false;
    }

    PimDestroyBo(host_opr1);
    PimDestroyBo(host_opr2);
    PimDestroyBo(host_out);
    PimDestroyBo(ref_out);
    PimDestroyBo(device_opr1);
    PimDestroyBo(device_opr2);
    PimDestroyBo(device_output);

    PimDeinitialize();
    return ret;
}

TEST(OCLPimIntegrationTest, PimEltWiseAddSync) { EXPECT_TRUE(pim_eltwise_add_sync()); }
