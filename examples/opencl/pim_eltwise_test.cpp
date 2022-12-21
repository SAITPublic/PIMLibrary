
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

using half_float::half;
using namespace std;

class PimEltwiseTest : public Testing
{
   public:
    PimEltwiseTest(unsigned in_length, string op) : n_(in_length), op_(op)
    {
        host_opr1_ = PimCreateBo(n_, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
        host_opr2_ = PimCreateBo(n_, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
        host_out_ = PimCreateBo(n_, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
        ref_out_ = PimCreateBo(n_, 1, 1, 1, PIM_FP16, MEM_TYPE_HOST);
        device_opr1_ = PimCreateBo(n_, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
        device_opr2_ = PimCreateBo(n_, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
        device_output_ = PimCreateBo(n_, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    }

    ~PimEltwiseTest(void)
    {
        PimDestroyBo(host_opr1_);
        PimDestroyBo(host_opr2_);
        PimDestroyBo(host_out_);
        PimDestroyBo(ref_out_);
        PimDestroyBo(device_opr1_);
        PimDestroyBo(device_opr2_);
        PimDestroyBo(device_output_);
    }

    virtual void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.2f) override
    {
        set_rand_half_data((half *)host_opr1_->data, (half)0.5, n_);
        set_rand_half_data((half *)host_opr2_->data, (half)0.5, n_);
        if (op_ == "add")
            addCPU((half *)host_opr1_->data, (half *)host_opr2_->data, (half *)ref_out_->data, n_);
        else
            mulCPU((half *)host_opr1_->data, (half *)host_opr2_->data, (half *)ref_out_->data, n_);
        PimCopyMemory(device_opr1_, host_opr1_, HOST_TO_PIM);
        PimCopyMemory(device_opr2_, host_opr2_, HOST_TO_PIM);
    }

    virtual void run(bool block = true, unsigned niter = 1) override
    {
        for (unsigned i = 0; i < niter; ++i) {
            if (op_ == "add")
                PimExecuteAdd(device_output_, device_opr1_, device_opr2_, nullptr, true);
            else
                PimExecuteMul(device_output_, device_opr1_, device_opr2_, nullptr, true);
            if (!block) PimSynchronize();
        }
        PimCopyMemory(host_out_, device_output_, PIM_TO_HOST);
    }

    virtual int validate(float epsilon = 0.1f) override
    {
        return compare_half_relative((half *)host_out_->data, (half *)ref_out_->data, n_);
    }

   private:
    unsigned n_;
    string op_;

    PimBo *host_opr1_, *host_opr2_, *host_out_;
    PimBo *device_opr1_, *device_opr2_, *device_output_;
    PimBo *ref_out_;
};

class PimEltwiseTestFixture : public ::testing::Test
{
   protected:
    void SetUp(PimRuntimeType plt)
    {
        PimInitialize(plt, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown(void) override { PimDeinitialize(); }
    int ExecuteTest(unsigned n, string op, bool block = true)
    {
        PimEltwiseTest pimEltwiseTest = PimEltwiseTest(n, op);
        pimEltwiseTest.prepare();
        pimEltwiseTest.run(block);
        return pimEltwiseTest.validate();
    }
};
// OpenCL Test Cases
TEST_F(PimEltwiseTestFixture, ocl_pim_eltWise_add_256x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(256 * 1024, "add") == 0);
}
TEST_F(PimEltwiseTestFixture, ocl_pim_eltWise_mul_256x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(256 * 1024, "mul") == 0);
}
