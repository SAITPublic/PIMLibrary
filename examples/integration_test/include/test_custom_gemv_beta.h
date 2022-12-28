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

class PimGemvTest : public Testing
{
   public:
    PimGemvTest(unsigned in_h, unsigned in_w, unsigned out_h, unsigned out_w, PimGemmOrder gemm_order)
        : in_h_(in_h), in_w_(in_w), out_h_(out_h), out_w_(out_w), gemm_order_(gemm_order)
    {
        desc_ = PimCreateGemmDesc(1, 1, in_h_, in_w_, out_h_, out_w_, PIM_FP16, gemm_order);
        if (gemm_order == I_X_W) {
            in_size_ = in_w_;
            out_size_ = out_w_;
        } else {
            in_size_ = in_h_;
            out_size_ = out_h_;
        }
        wgt_size_ = in_size_ * out_size_;
        h_i_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_INPUT);
        h_w_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_WEIGHT);
        h_o_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_OUTPUT);
        d_i_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_INPUT);
        d_w_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_WEIGHT);
        d_o_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_OUTPUT);
        golden_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_OUTPUT);
    }

    ~PimGemvTest(void)
    {
        PimDestroyBo(h_i_);
        PimDestroyBo(h_w_);
        PimDestroyBo(h_o_);
        PimDestroyBo(golden_);
        PimDestroyBo(d_i_);
        PimDestroyBo(d_w_);
        PimDestroyBo(d_o_);
        PimDestroyGemmDesc(desc_);
    }

    virtual void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.1f) override
    {
        set_rand_half_data((half*)h_i_->data, half(variation), in_size_);
        set_rand_half_data((half*)h_w_->data, half(variation), wgt_size_);
        set_half_data((half*)golden_->data, half(0.0), out_size_);

        half* h_i_data = (half*)h_i_->data;
        half* h_w_data = (half*)h_w_->data;
        half* golden_data = (half*)golden_->data;

        if (gemm_order_ == I_X_W) {
            matmulCPU(h_i_data, h_w_data, golden_data, in_h_, out_w_, in_w_, half(alpha), half(beta));
        } else {
            matmulCPU(h_w_data, h_i_data, golden_data, out_h_, out_w_, in_h_, half(alpha), half(beta));
        }
        PimCopyMemory(d_i_, h_i_, HOST_TO_DEVICE);
        PimCopyMemory(d_w_, h_w_, HOST_TO_DEVICE);
        PimCopyMemory(d_o_, h_o_, HOST_TO_DEVICE);
    }

    virtual void run(bool block = true, unsigned niter = 1) override
    {
        for (unsigned i = 0; i < niter; ++i) {
            (void)PimExecuteGemm(d_o_, d_i_, d_w_, d_o_, PimActFunc::NONE, gemm_order_, nullptr, block);
            if (!block) PimSynchronize();
        }
        PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST);
    }

    virtual int validate(float epsilon = 0.1f) override
    {
        return compare_half_relative((half*)golden_->data, (half*)h_o_->data, out_size_, epsilon);
    }

   private:
    unsigned in_h_;
    unsigned in_w_;
    unsigned out_h_;
    unsigned out_w_;

    PimGemmOrder gemm_order_;

    unsigned in_size_;
    unsigned wgt_size_;
    unsigned out_size_;

    PimGemmDesc* desc_;
    PimBo *h_i_, *h_w_, *h_o_;  // input, weight, bias, output
    PimBo *d_i_, *d_w_, *d_o_;
    PimBo* golden_;
};

class PimGemvTestFixture : public ::testing::Test
{
   protected:
    void SetUp(PimRuntimeType plt)
    {
        PimInitialize(plt, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown(void) override { PimDeinitialize(); }
    int ExecuteTest(unsigned in_h, unsigned in_w, unsigned out_h, unsigned out_w, PimGemmOrder gemm_order = I_X_W,
                    bool block = true)
    {
        PimGemvTest pimGemvTest = PimGemvTest(in_h, in_w, out_h, out_w, gemm_order);
        pimGemvTest.prepare();
        pimGemvTest.run(block);
        return pimGemvTest.validate();
    }
};
