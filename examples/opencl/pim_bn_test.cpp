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
#include <cmath>
#include <iostream>
#include "half.hpp"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_profile.h"
#include "utility/test_util.h"

#ifdef DEBUG_PIM
#define NUM_ITER (1)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

inline float convertH2F(half h_val) { return half_float::detail::half2float<float>(h_val); }
void calculate_bn(half* input, half* output, half* mean, half* variance, half* gamma, half* beta, int len)
{
    for (int i = 0; i < len; i++) {
        output[i] = gamma[0] * ((input[i] - mean[0]) / sqrt(variance[0])) + beta[0];
    }
}

class PimBNTest : public Testing
{
   public:
    PimBNTest(unsigned in_length) : n_(in_length)
    {
        PimDesc* pim_desc = PimCreateDesc(BATCH_, CH_, HEIGHT_, n_, PIM_FP16);
        host_beta_ = PimCreateBo(1, CH_, 1, 1, PIM_FP16, MEM_TYPE_HOST);
        host_gamma_ = PimCreateBo(1, CH_, 1, 1, PIM_FP16, MEM_TYPE_HOST);
        host_mean_ = PimCreateBo(1, CH_, 1, 1, PIM_FP16, MEM_TYPE_HOST);
        host_variance_ = PimCreateBo(1, CH_, 1, 1, PIM_FP16, MEM_TYPE_HOST);

        /* __PIM_API__ call : Create PIM Buffer Object */
        host_input_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        host_output_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        golden_output_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        pim_input_ = PimCreateBo(pim_desc, MEM_TYPE_PIM);
        device_output_ = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    }

    ~PimBNTest(void)
    {
        PimDestroyBo(host_input_);
        PimDestroyBo(host_beta_);
        PimDestroyBo(host_gamma_);
        PimDestroyBo(host_mean_);
        PimDestroyBo(host_variance_);
        PimDestroyBo(host_output_);
        PimDestroyBo(golden_output_);
        PimDestroyBo(device_output_);
        PimDestroyBo(pim_input_);
    }

    virtual void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.2f) override
    {
        set_rand_half_data((half*)host_input_->data, (half)0.5, n_);
        set_rand_half_data_positive((half*)host_beta_->data, (half)0.5, CH_);
        set_rand_half_data_positive((half*)host_gamma_->data, (half)0.5, CH_);
        set_rand_half_data_positive((half*)host_mean_->data, (half)0.5, CH_);
        set_rand_half_data_positive((half*)host_variance_->data, (half)0.5, CH_);
        set_rand_half_data((half*)golden_output_->data, (half)0.5, n_);
        calculate_bn((half*)host_input_->data, (half*)golden_output_->data, (half*)host_mean_->data,
                     (half*)host_variance_->data, (half*)host_gamma_->data, (half*)host_beta_->data, n_);

        PimCopyMemory(pim_input_, host_input_, HOST_TO_PIM);
    }

    virtual void run(bool block = true, unsigned niter = 1) override
    {
        for (unsigned i = 0; i < niter; ++i) {
            PimExecuteBN(device_output_, pim_input_, host_beta_, host_gamma_, host_mean_, host_variance_, 1e-5, nullptr,
                         block);
        }
        PimCopyMemory(host_output_, device_output_, PIM_TO_HOST);
    }

    virtual int validate(float epsilon = 0.1f) override
    {
        return compare_half_relative((half*)host_output_->data, (half*)golden_output_->data, n_);
    }

   private:
    const int BATCH_ = 1;
    const int CH_ = 1;
    const int HEIGHT_ = 1;
    unsigned n_;

    PimBo *host_input_, *host_beta_, *host_gamma_, *host_mean_, *host_variance_, *host_output_;
    PimBo *pim_input_, *device_output_;
    PimBo* golden_output_;
};

class PimBNTestFixture : public ::testing::Test
{
   protected:
    void SetUp(PimRuntimeType plt)
    {
        PimInitialize(plt, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown(void) override { PimDeinitialize(); }
    int ExecuteTest(unsigned n, bool block = true)
    {
        PimBNTest pimBNTest = PimBNTest(n);
        pimBNTest.prepare();
        pimBNTest.run(block);
        return pimBNTest.validate();
    }
};
// OpenCL Test Cases
TEST_F(PimBNTestFixture, ocl_pim_BN_1x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1 * 1024) == 0);
}
TEST_F(PimBNTestFixture, ocl_pim_BN_64x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(64 * 1024) == 0);
}
TEST_F(PimBNTestFixture, ocl_pim_BN_128x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(128 * 1024) == 0);
}
