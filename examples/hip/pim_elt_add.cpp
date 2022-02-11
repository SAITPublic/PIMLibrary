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
#include "hip/hip_runtime.h"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_profile.h"

#ifdef DEBUG_PIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

int pim_elt_add_up_to_512KB(bool block, uint32_t input_len)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, input_len, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input0 = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_input1 = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* pim_input0 = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* pim_input1 = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    std::string test_vector_data = TEST_VECTORS_DATA;
    /* Initialize the input, weight, output data */
    std::string input0 = test_vector_data + "load/elt_add/input0_512KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_512KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_512KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_add/output_512KB.dat";

    /* Initialize the input, weight, output data */
    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input0, host_input0, HOST_TO_PIM);
    PimCopyMemory(pim_input1, host_input1, HOST_TO_PIM);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (ELT_ADD) */
        PimExecuteAdd(device_output, pim_input0, pim_input1, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, PIM_TO_HOST);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, input_len);
    }
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input0);
    PimDestroyBo(host_input1);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input0);
    PimDestroyBo(pim_input1);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_elt_add_profile(bool block, int len)
{
    int ret = 0;
    int length = len;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input0 = PimCreateBo(length, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_input1 = PimCreateBo(length, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(length, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(length, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);

    PimBo* pim_input1 = PimCreateBo(length, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(length, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    PimBo* pim_input0 = PimCreateBo(length, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/elt_add/input0_32768KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_32768KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_32768KB.dat";
    //    std::string output_dump = test_vector_data + "dump/elt_add/output_32768KB.dat";

    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input0, host_input0, HOST_TO_PIM);
    PimCopyMemory(pim_input1, host_input1, HOST_TO_PIM);

    //    PimExecuteDummy();
    PimExecuteAdd(device_output, pim_input0, pim_input1, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);
/* __PIM_API__ call : Execute PIM kernel (ELT_ADD) */
#ifndef EMULATOR
    int iter;
    //    PIM_PROFILE_TICK(ELT_ADD_1);
    for (iter = 0; iter < 100; iter++) {
#endif
        PimExecuteAdd(device_output, pim_input0, pim_input1, nullptr, block);
#ifndef EMULATOR
    }
    //    if (!block) PimSynchronize();
    //    PIM_PROFILE_TOCK(ELT_ADD_1);
    //    printf("[ %d execution time ]\n", iter);
    float eventMs = 0.0f;

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);
#endif
    PimCopyMemory(host_output, device_output, PIM_TO_HOST);

    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, host_output->size / sizeof(half));

    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input0);
    PimDestroyBo(host_input1);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input0);
    PimDestroyBo(pim_input1);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, PimEltAdd1Sync) { EXPECT_TRUE(pim_elt_add_up_to_512KB(true, 1 * 1024) == 0); }
TEST(HIPIntegrationTest, PimEltAdd1Async) { EXPECT_TRUE(pim_elt_add_up_to_512KB(false, 1 * 1024) == 0); }
TEST(HIPIntegrationTest, PimEltAdd2Sync) { EXPECT_TRUE(pim_elt_add_up_to_512KB(true, 128 * 1024) == 0); }
TEST(HIPIntegrationTest, PimEltAdd2ASync) { EXPECT_TRUE(pim_elt_add_up_to_512KB(false, 128 * 1024) == 0); }
TEST(HIPIntegrationTest, PimEltAdd3Sync) { EXPECT_TRUE(pim_elt_add_up_to_512KB(true, 256 * 1024) == 0); }
TEST(HIPIntegrationTest, PimEltAdd3Async) { EXPECT_TRUE(pim_elt_add_up_to_512KB(false, 256 * 1024) == 0); }
TEST(HIPIntegrationTest, PimEltAdd4Sync) { EXPECT_TRUE(pim_elt_add_up_to_512KB(true, 128 * 768) == 0); }
TEST(HIPIntegrationTest, PimEltAdd4ASync) { EXPECT_TRUE(pim_elt_add_up_to_512KB(false, 128 * 768) == 0); }
TEST(HIPIntegrationTest, PimEltAddProfile1Sync) { EXPECT_TRUE(pim_elt_add_profile(true, (128 * 1024)) == 0); }
TEST(HIPIntegrationTest, PimEltAddProfile1Async) { EXPECT_TRUE(pim_elt_add_profile(false, (128 * 1024)) == 0); }
// TEST(HIPIntegrationTest, PimEltAddProfile2Async) { EXPECT_TRUE(pim_elt_add_profile(false, (256 * 1024)) == 0); }
// TEST(HIPIntegrationTest, PimEltAddProfile3Async) { EXPECT_TRUE(pim_elt_add_profile(false, (512 * 1024)) == 0); }
// TEST(HIPIntegrationTest, PimEltAddProfile4Async) { EXPECT_TRUE(pim_elt_add_profile(false, (1024 * 1024)) == 0); }
// TEST(HIPIntegrationTest, PimEltAddProfile5Async) { EXPECT_TRUE(pim_elt_add_profile(false, (2048 * 1024)) == 0); }
// TEST(HIPIntegrationTest, PimEltAddProfile6Async) { EXPECT_TRUE(pim_elt_add_profile(false, (4096 * 1024)) == 0); }
// TEST(HIPIntegrationTest, PimEltAddProfile7Async) { EXPECT_TRUE(pim_elt_add_profile(false, (8192 * 1024)) == 0); }
// TEST(HIPIntegrationTest, PimEltAddProfile8Async) { EXPECT_TRUE(pim_elt_add_profile(false, (16384 * 1024)) == 0); }
