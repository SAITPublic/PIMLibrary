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

using half_float::half;
using namespace std;

class OclPimGemmTest
{
   public:
    OclPimGemmTest(unsigned n, unsigned c, unsigned h, unsigned in_w, unsigned out_w, PimActFunc act, bool has_bias)
        : n_(n), c_(c), h_(h), in_w_(in_w), out_w_(out_w), act_(act), has_bias_(has_bias)
    {
        if (!is_support_activation(act_)) {
            throw invalid_argument("Invalid activation type");
        }

        in_size_ = n_ * c_ * h_ * in_w_;
        wgt_size_ = n_ * c_ * in_w_ * out_w_;
        out_size_ = n_ * c_ * h_ * out_w_;

        desc_ = PimCreateGemmDesc(n_, c_, h_, in_w_, out_w_, PIM_FP16);
        h_i_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_INPUT);
        h_w_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_WEIGHT);
        h_b_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_BIAS);
        h_o_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_OUTPUT);
        d_i_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_INPUT);
        d_w_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_WEIGHT);
        if (has_bias_)
            d_b_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_BIAS);
        else
            d_b_ = nullptr;

        d_o_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_OUTPUT);
        golden_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_OUTPUT);
    }

    OclPimGemmTest(unsigned n_, unsigned c_, unsigned h_, unsigned in_w_, unsigned out_w_)
        : OclPimGemmTest(n_, c_, h_, in_w_, out_w_, ACT_RELU, true)
    {
    }

    ~OclPimGemmTest()
    {
        PimDestroyBo(h_i_);
        PimDestroyBo(h_w_);
        PimDestroyBo(h_b_);
        PimDestroyBo(h_o_);
        PimDestroyBo(golden_);
        PimDestroyBo(d_i_);
        PimDestroyBo(d_w_);
        if (has_bias_) PimDestroyBo(d_b_);
        PimDestroyBo(d_o_);
    }

    void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.01f)
    {
        set_half_data((half*)golden_->data, half(0.0), out_size_);
        set_half_data((half*)h_o_->data, half(0.0), out_size_);
        set_rand_half_data((half*)h_i_->data, half(variation), in_size_);
        set_rand_half_data((half*)h_w_->data, half(variation), wgt_size_);
        if (has_bias_) set_rand_half_data((half*)h_b_->data, half(variation), out_size_);

        half* h_i_data = (half*)h_i_->data;
        half* h_w_data = (half*)h_w_->data;
        half* golden_data = (half*)golden_->data;
        for (int nc_i = 0; nc_i < n_ * c_; nc_i++) {
            matmulCPU(h_i_data, h_w_data, golden_data, h_, out_w_, in_w_, half(alpha), half(beta));
            h_i_data += (in_w_ * h_);
            h_w_data += (in_w_ * out_w_);
            golden_data += (out_w_ * h_);
        }
        if (has_bias_) {
            addBiasCPU((half*)golden_->data, (half*)h_b_->data, out_size_);
        }
        if (act_ == ACT_RELU) {
            reluCPU((half*)golden_->data, out_size_);
        }
        PimCopyMemory(d_i_, h_i_, HOST_TO_DEVICE);
        PimCopyMemory(d_w_, h_w_, HOST_TO_DEVICE);
        PimCopyMemory(d_o_, h_o_, HOST_TO_DEVICE);

        if (has_bias_) {
            PimCopyMemory(d_b_, h_b_, HOST_TO_DEVICE);
        }
    }

    void run(bool block = true, unsigned niter = 1)
    {
        for (unsigned i = 0; i < niter; ++i) {
            (void)PimExecuteGemm(d_o_, d_i_, d_w_, has_bias_ ? d_b_ : nullptr, act_, nullptr, block);
            if (!block) PimSynchronize();
        }
        PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST);
    }

    void run_with_explicit_reordering(bool use_device_weight, bool block = true, unsigned niter = 1)
    {
        auto* w_to_reorder = use_device_weight ? d_w_ : h_w_;
        for (unsigned i = 0; i < niter; ++i) {
            auto* reordered_pim_w = PimConvertGemmWeight(w_to_reorder);
            // Ignore return value here to avoid extra branches.
            // Please check the success of the API call in logs.
            // Results are verified further.
            (void)PimExecuteGemm(d_o_, d_i_, reordered_pim_w, d_b_, act_, nullptr, block);
            if (!block) PimSynchronize();
            PimDestroyBo(reordered_pim_w);
        }
        PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST);
    }

    int validate(float epsilon = 1e-2)
    {
        return compare_half_relative((half*)h_o_->data, (half*)golden_->data, out_size_, epsilon);
    }

   private:
    bool is_support_activation(const PimActFunc& act) { return (act == ACT_RELU || act == NONE) ? true : false; }
    // (n_, c, h, in_w) * (n_, c, in_w, out_w_) = (n_, c, h, out_w_)
    unsigned n_;
    unsigned c_;
    unsigned h_;
    unsigned in_w_;
    unsigned out_w_;

    PimActFunc act_;
    bool has_bias_;

    unsigned in_size_;
    unsigned wgt_size_;
    unsigned out_size_;

    PimGemmDesc* desc_;
    PimBo *h_i_, *h_w_, *h_b_, *h_o_;  // input, weight, bias, output
    PimBo *d_i_, *d_w_, *d_b_, *d_o_;
    PimBo* golden_;
};

class PimGemmOCLTestFixture : public ::testing::Test
{
   protected:
    virtual void SetUp() override
    {
        PimInitialize(RT_TYPE_OPENCL, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown() override { PimDeinitialize(); }
    int ExecuteTest(unsigned n, unsigned c, unsigned h, unsigned in_w, unsigned out_w, bool has_bias = true,
                    bool block = true, PimActFunc act = ACT_RELU)
    {
        OclPimGemmTest pimGemmTest = OclPimGemmTest(n, c, h, in_w, out_w, act, has_bias);
        pimGemmTest.prepare();
        pimGemmTest.run(block);
        return pimGemmTest.validate();
    }
    int ExecuteTestExplicitReordering(unsigned n, unsigned c, unsigned h, unsigned in_w, unsigned out_w,
                                      bool use_device_weight, bool has_bias = true, bool block = true,
                                      PimActFunc act = ACT_RELU)
    {
        OclPimGemmTest pimGemmTest = OclPimGemmTest(n, c, h, in_w, out_w, act, has_bias);
        pimGemmTest.prepare();
        pimGemmTest.run_with_explicit_reordering(use_device_weight, block);
        return pimGemmTest.validate();
    }
};

TEST_F(PimGemmOCLTestFixture, pim_gemm_1x1024_1024x4096)
{
    EXPECT_TRUE(ExecuteTest(1, 1, 1, 1024, 4096, false, true, NONE) == 0);
}