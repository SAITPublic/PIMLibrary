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
#include "utility/pim_debug.hpp"

#define LENGTH (128 * 1024)

using namespace std;
using half_float::half;

int pim_sv_add_up_to_256KB(uint32_t input_len)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, input_len, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_scalar = PimCreateBo(1, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_vector = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* pim_vector = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/sv_add/scalar_2B.dat";
    std::string input1 = test_vector_data + "load/sv_add/vector_256KB.dat";
    std::string output = test_vector_data + "load/sv_add/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/sv_add/output_256KB.dat";

    load_data(input0.c_str(), (char*)host_scalar->data, host_scalar->size);
    load_data(input1.c_str(), (char*)host_vector->data, host_vector->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_vector, host_vector, HOST_TO_PIM);
    /* __PIM_API__ call : Execute PIM kernel (ELT_ADD) */
    PimExecuteAdd(device_output, host_scalar->data, pim_vector);

    PimCopyMemory(host_output, device_output, PIM_TO_HOST);

    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, input_len);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_scalar);
    PimDestroyBo(host_vector);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_vector);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_sv_mul_up_to_256KB(uint32_t input_len)
{
    int ret = 0;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, input_len, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_scalar = PimCreateBo(1, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_vector = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* pim_vector = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/sv_mul/scalar_2B.dat";
    std::string input1 = test_vector_data + "load/sv_mul/vector_256KB.dat";
    std::string output = test_vector_data + "load/sv_mul/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/sv_mul/output_256KB.dat";

    load_data(input0.c_str(), (char*)host_scalar->data, host_scalar->size);
    load_data(input1.c_str(), (char*)host_vector->data, host_vector->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_vector, host_vector, HOST_TO_PIM);

    /* __PIM_API__ call : Execute PIM kernel (ELT_ADD) */
    PimExecuteMul(device_output, host_scalar->data, pim_vector);

    PimCopyMemory(host_output, device_output, PIM_TO_HOST);

    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, input_len);
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_scalar);
    PimDestroyBo(host_vector);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_vector);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

// TEST(HIPIntegrationTest, PimSVAdd1) { EXPECT_TRUE(pim_sv_add_up_to_256KB(1 * 1024) == 0); }
// TEST(HIPIntegrationTest, PimSVAdd2) { EXPECT_TRUE(pim_sv_add_up_to_256KB(10 * 1024) == 0); }
// TEST(HIPIntegrationTest, PimSVAdd4) { EXPECT_TRUE(pim_sv_add_up_to_256KB(128 * 1024) == 0); }

// TEST(HIPIntegrationTest, PimSVMul1) { EXPECT_TRUE(pim_sv_mul_up_to_256KB(1 * 1024) == 0); }
// TEST(HIPIntegrationTest, PimSVMul2) { EXPECT_TRUE(pim_sv_mul_up_to_256KB(10 * 1024) == 0); }
// TEST(HIPIntegrationTest, PimSVMul4) { EXPECT_TRUE(pim_sv_mul_up_to_256KB(128 * 1024) == 0); }
