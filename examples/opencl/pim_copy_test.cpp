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
#include "utility/pim_profile.h"
#include "utility/test_util.h"

#ifdef DEBUG_PIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

class PimCopyTest : public Testing
{
   public:
    PimCopyTest(unsigned in_length) : n_(in_length)
    {
        PimDesc* pim_desc = PimCreateDesc(1, 1, 1, n_, PIM_FP16);

        /* __PIM_API__ call : Create PIM Buffer Object */
        host_input_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        host_output_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        golden_output_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        pim_input_ = PimCreateBo(pim_desc, MEM_TYPE_PIM);
        device_output_ = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    }

    ~PimCopyTest(void)
    {
        PimDestroyBo(host_input_);
        PimDestroyBo(host_output_);
        PimDestroyBo(golden_output_);
        PimDestroyBo(device_output_);
        PimDestroyBo(pim_input_);
    }

    virtual void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.2f) override
    {
        set_rand_half_data((half_float::half*)host_input_->data, (half_float::half)0.5, n_);
        /* __PIM_API__ call : Preload weight data on PIM memory */
        PimCopyMemory(pim_input_, host_input_, HOST_TO_PIM);
    }

    virtual void run(bool block = true, unsigned niter = 1) override
    {
        for (unsigned i = 0; i < niter; ++i) {
            PimCopyMemory(device_output_, pim_input_, PIM_TO_PIM);
            if (!block) PimSynchronize();
            PimCopyMemory(host_output_, device_output_, PIM_TO_HOST);
        }
    }

    virtual int validate(float epsilon = 0.1f) override
    {
        return compare_half_relative((half*)host_output_->data, (half*)host_input_->data, n_);
    }

   private:
    unsigned n_;

    PimBo *host_input_, *host_output_;
    PimBo *pim_input_, *device_output_;
    PimBo* golden_output_;
};

class PimCopyTestFixture : public ::testing::Test
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
        PimCopyTest pimCopyTest = PimCopyTest(n);
        pimCopyTest.prepare();
        pimCopyTest.run(block);
        return pimCopyTest.validate();
    }
};
// OpenCL Test Cases
TEST_F(PimCopyTestFixture, ocl_pim_copy_128x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1 * 1024) == 0);
}
