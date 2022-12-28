/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "utility/pim_debug.hpp"
#include "utility/pim_profile.h"
#include "half.hpp"
#include "pim_runtime_api.h"
#include "test_util.h"

using half_float::half;
using namespace std;

class PimEltwiseTest : public Testing
{
   public:
    PimEltwiseTest(unsigned in_length, unsigned c, unsigned h, unsigned w, string op)
        : n_(in_length), c_(c), h_(h), w_(w), op_(op)
    {
        host_opr1_ = PimCreateBo(n_, c_, h_, w_, PIM_FP16, MEM_TYPE_HOST);
        host_opr2_ = PimCreateBo(n_, c_, h_, w_, PIM_FP16, MEM_TYPE_HOST);
        host_out_ = PimCreateBo(n_, c_, h_, w_, PIM_FP16, MEM_TYPE_HOST);
        ref_out_ = PimCreateBo(n_, c_, h_, w_, PIM_FP16, MEM_TYPE_HOST);
        device_opr1_ = PimCreateBo(n_, c_, h_, w_, PIM_FP16, MEM_TYPE_PIM);
        device_opr2_ = PimCreateBo(n_, c_, h_, w_, PIM_FP16, MEM_TYPE_PIM);
        device_output_ = PimCreateBo(n_, c_, h_, w_, PIM_FP16, MEM_TYPE_PIM);
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
            addCPU((half *)host_opr1_->data, (half *)host_opr2_->data, (half *)ref_out_->data, n_*c_*h_*w_);
        else
            mulCPU((half *)host_opr1_->data, (half *)host_opr2_->data, (half *)ref_out_->data, n_*c_*h_*w_);
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
    unsigned n_, c_, h_, w_;
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
    int ExecuteTest(unsigned n, unsigned c, unsigned h, unsigned w, string op, bool block = true)
    {
        PimEltwiseTest pimEltwiseTest = PimEltwiseTest(n, c, h, w, op);
        pimEltwiseTest.prepare();
        pimEltwiseTest.run(block);
        return pimEltwiseTest.validate();
    }
};
