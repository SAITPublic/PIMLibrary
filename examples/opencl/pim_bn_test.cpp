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

#ifdef DEBUG_PIM
#define NUM_ITER (1)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

inline float convertH2F(half h_val) { return half_float::detail::half2float<float>(h_val); }
int pim_bn_up_to_256KB(bool block, uint64_t input_len)
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int HEIGHT = 1;
    const int WIDTH = input_len;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_OPENCL, PIM_FP16);

    PimDesc* pim_desc = PimCreateDesc(BATCH, CH, HEIGHT, WIDTH, PIM_FP16);

    PimBo* host_beta = PimCreateBo(1, CH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, CH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, CH, 1, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_variance = PimCreateBo(1, CH, 1, 1, PIM_FP16, MEM_TYPE_HOST);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* pim_input = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/bn/nr_input_256KB.dat";
    std::string beta = test_vector_data + "load/bn/nr_beta_256KB.dat";
    std::string gamma = test_vector_data + "load/bn/nr_gamma_256KB.dat";
    std::string mean = test_vector_data + "load/bn/nr_mean_256KB.dat";
    std::string variance = test_vector_data + "load/bn/nr_variance_256KB.dat";
    std::string output = test_vector_data + "load/bn/nr_output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/bn/nr_output_256KB.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(beta.c_str(), (char*)host_beta->data, host_beta->size);
    load_data(gamma.c_str(), (char*)host_gamma->data, host_gamma->size);
    load_data(mean.c_str(), (char*)host_mean->data, host_mean->size);
    load_data(variance.c_str(), (char*)host_variance->data, host_variance->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    // /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);
    // /* __PIM_API__ call : Execute PIM kernel */
    for (int i = 0; i < NUM_ITER; i++) {
        PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_variance, 1e-5, nullptr, block);
    }

    PimCopyMemory(host_output, device_output, PIM_TO_HOST);

    ret = compare_half_relative((half*)host_output->data, (half*)golden_output->data, WIDTH);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_variance);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

TEST(OCLPimIntegrationTest, PimNRBN1) { EXPECT_TRUE(pim_bn_up_to_256KB(true, 1 * 1024) == 0); }
TEST(OCLPimIntegrationTest, PimNRBN2) { EXPECT_TRUE(pim_bn_up_to_256KB(true, 64 * 1024) == 0); }
TEST(OCLPimIntegrationTest, PimNRBN3) { EXPECT_TRUE(pim_bn_up_to_256KB(true, 128 * 1024) == 0); }
