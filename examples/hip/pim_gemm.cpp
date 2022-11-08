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
#include "half.hpp"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_profile.h"

using half_float::half;
using namespace std;

class PimGemmTest
{
   public:
    PimGemmTest(unsigned n, unsigned c, unsigned in_h, unsigned in_w, unsigned out_h, unsigned out_w, PimActFunc act,
                bool has_bias, PimGemmOrder gemm_order)
        : n_(n),
          c_(c),
          in_h_(in_h),
          in_w_(in_w),
          out_h_(out_h),
          out_w_(out_w),
          act_(act),
          has_bias_(has_bias),
          gemm_order_(gemm_order)
    {
        if (!is_support_activation(act_)) {
            throw invalid_argument("Invalid activation type");
        }

        in_size_ = n_ * c_ * in_h_ * in_w_;
        out_size_ = n_ * c_ * out_h_ * out_w_;
        if (gemm_order_ == W_X_I)
            wgt_size_ = n_ * c_ * in_h_ * out_h_;
        else
            wgt_size_ = n_ * c_ * in_w_ * out_w_;
        desc_ = PimCreateGemmDesc(n_, c_, in_h_, in_w_, out_h_, out_w_, PIM_FP16, gemm_order_);
        h_i_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_INPUT);
        h_w_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_WEIGHT);
        if (has_bias_) h_b_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_BIAS);
        h_o_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_OUTPUT);
        d_i_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_INPUT);
        d_w_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_WEIGHT);
        if (has_bias_) d_b_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_BIAS);
        d_o_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_OUTPUT);
        golden_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_OUTPUT);
    }

    ~PimGemmTest(void)
    {
        PimDestroyBo(h_i_);
        PimDestroyBo(h_w_);
        if (has_bias_) PimDestroyBo(h_b_);
        PimDestroyBo(h_o_);
        PimDestroyBo(golden_);
        PimDestroyBo(d_i_);
        PimDestroyBo(d_w_);
        if (has_bias_) PimDestroyBo(d_b_);
        PimDestroyBo(d_o_);
        PimDestroyGemmDesc(desc_);
    }

    void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.2f)
    {
        set_half_data((half*)golden_->data, half(0.0), out_size_);
        set_half_data((half*)h_o_->data, half(0.0), out_size_);
        set_rand_half_data((half*)h_i_->data, half(variation), in_size_);
        set_rand_half_data((half*)h_w_->data, half(variation), wgt_size_);
        if (has_bias_) set_rand_half_data((half*)h_b_->data, half(variation), out_size_);

        half* h_i_data = (half*)h_i_->data;
        half* h_w_data = (half*)h_w_->data;
        half* golden_data = (half*)golden_->data;

        if (gemm_order_ == W_X_I) {
            for (int nc_i = 0; nc_i < n_ * c_; nc_i++) {
                matmulCPU(h_w_data, h_i_data, golden_data, out_h_, out_w_, in_h_, half(alpha), half(beta));
                h_i_data += (in_h_ * in_w_);
                h_w_data += (in_h_ * out_h_);
                golden_data += (out_h_ * out_w_);
            }
        } else {
            for (int nc_i = 0; nc_i < n_ * c_; nc_i++) {
                matmulCPU(h_i_data, h_w_data, golden_data, in_h_, out_w_, in_w_, half(alpha), half(beta));
                h_i_data += (in_h_ * in_w_);
                h_w_data += (in_w_ * out_w_);
                golden_data += (out_h_ * out_w_);
            }
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
        } else {
            d_b_ = nullptr;
        }
    }

    void run(bool block = true, unsigned niter = 1)
    {
        for (unsigned i = 0; i < niter; ++i) {
            (void)PimExecuteGemm(d_o_, d_i_, d_w_, d_b_, act_, gemm_order_, nullptr, block);
            if (!block) PimSynchronize();
        }
        PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST);
    }

    void run_with_explicit_reordering(bool use_device_weight, bool block = true, unsigned niter = 1)
    {
        auto* w_to_reorder = use_device_weight ? d_w_ : h_w_;
        for (unsigned i = 0; i < niter; ++i) {
            auto* reordered_pim_w = PimConvertGemmWeight(w_to_reorder, gemm_order_);
            // Ignore return value here to avoid extra branches.
            // Please check the success of the API call in logs.
            // Results are verified further.
            (void)PimExecuteGemm(d_o_, d_i_, reordered_pim_w, d_b_, act_, gemm_order_, nullptr, block);
            if (!block) PimSynchronize();
            PimDestroyBo(reordered_pim_w);
        }
        PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST);
    }

    void run_with_explicit_reordering_on_device(bool use_device_weight, bool block = true, unsigned niter = 1)
    {
        auto* w_to_reorder = use_device_weight ? d_w_ : h_w_;
        for (unsigned i = 0; i < niter; ++i) {
            auto* reordered_pim_w =
                PimConvertGemmWeight(w_to_reorder, gemm_order_, /*reorder_on_device = */ true, nullptr);
            // Ignore return value here to avoid extra branches.
            // Please check the success of the API call in logs.
            // Results are verified further.
            (void)PimExecuteGemm(d_o_, d_i_, reordered_pim_w, d_b_, act_, gemm_order_, nullptr, block);
            if (!block) PimSynchronize();
            PimDestroyBo(reordered_pim_w);
        }
        PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST);
    }

    int validate(float epsilon = 0.1f)
    {
        return compare_half_relative((half*)golden_->data, (half*)h_o_->data, out_size_, epsilon);
    }

   private:
    bool is_support_activation(const PimActFunc& act) { return (act == ACT_RELU || act == NONE) ? true : false; }

    unsigned n_;
    unsigned c_;
    unsigned in_h_;
    unsigned in_w_;
    unsigned out_h_;
    unsigned out_w_;

    PimActFunc act_;
    bool has_bias_;
    PimGemmOrder gemm_order_;

    unsigned in_size_;
    unsigned wgt_size_;
    unsigned out_size_;

    PimGemmDesc* desc_;
    PimBo *h_i_, *h_w_, *h_b_, *h_o_;  // input, weight, bias, output
    PimBo *d_i_, *d_w_, *d_b_, *d_o_;
    PimBo* golden_;
};

class PimGemmTestFixture : public ::testing::Test
{
   protected:
    virtual void SetUp(void) override
    {
        PimInitialize(RT_TYPE_HIP, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown(void) override { PimDeinitialize(); }

    int ExecuteTest(unsigned n, unsigned c, unsigned in_h, unsigned in_w, unsigned out_h, unsigned out_w,
                    PimGemmOrder gemm_order = I_X_W, bool has_bias = false, bool block = true, PimActFunc act = NONE)
    {
        PimGemmTest pimGemmTest = PimGemmTest(n, c, in_h, in_w, out_h, out_w, act, has_bias, gemm_order);
        pimGemmTest.prepare();
        pimGemmTest.run(block);
        return pimGemmTest.validate();
    }

    int ExecuteTestExplicitReordering(unsigned n, unsigned c, unsigned in_h, unsigned in_w, unsigned out_h,
                                      unsigned out_w, bool use_device_weight, PimGemmOrder gemm_order = I_X_W,
                                      bool has_bias = true, bool block = true, PimActFunc act = NONE)
    {
        PimGemmTest pimGemmTest = PimGemmTest(n, c, in_h, in_w, out_h, out_w, act, has_bias, gemm_order);
        pimGemmTest.prepare();
        pimGemmTest.run_with_explicit_reordering(use_device_weight, block);
        return pimGemmTest.validate();
    }

    int ExecuteTestExplicitReorderingOnDevice(unsigned n, unsigned c, unsigned in_h, unsigned in_w, unsigned out_h,
                                              unsigned out_w, bool use_device_weight, PimGemmOrder gemm_order = I_X_W,
                                              bool has_bias = true, bool block = true, PimActFunc act = NONE)
    {
        PimGemmTest pimGemmTest = PimGemmTest(n, c, in_h, in_w, out_h, out_w, act, has_bias, gemm_order);
        pimGemmTest.prepare();
        pimGemmTest.run_with_explicit_reordering_on_device(use_device_weight, block);
        return pimGemmTest.validate();
    }
};

TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_w_x_i_4096x1024_1024x1)
{
    EXPECT_TRUE(ExecuteTest(1, 1, 1024, 1, 4096, 1, W_X_I) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_w_x_i_2x4096x1024_2x1024x1)
{
    EXPECT_TRUE(ExecuteTest(1, 2, 1024, 1, 4096, 1, W_X_I) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_1x1024_1024x4096)
{
    EXPECT_TRUE(ExecuteTest(1, 1, 1, 1024, 1, 4096) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1024_1024x4096)
{
    EXPECT_TRUE(ExecuteTest(1, 1, 2, 1024, 2, 4096) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1x1024_2x1024x4096)
{
    EXPECT_TRUE(ExecuteTest(1, 2, 1, 1024, 1, 4096) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x2x1024_2x1024x4096)
{
    EXPECT_TRUE(ExecuteTest(1, 2, 2, 1024, 2, 4096) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x256_64x256x64)
{
    EXPECT_TRUE(ExecuteTest(1, 64, 1, 256, 1, 64) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x1024_64x1024x64)
{
    EXPECT_TRUE(ExecuteTest(1, 64, 1, 1024, 1, 64) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_4x1x4096_4x4096x1024)
{
    EXPECT_TRUE(ExecuteTest(1, 4, 1, 4096, 1, 1024) == 0);
}
#if 0
/* TODO:check */

TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_w_x_i_4096x240_240x1)
{
    EXPECT_TRUE(ExecuteTest(1, 1, 240, 1, 4096, 1, W_X_I) == 0);
}

TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_w_x_i_4096x1024_1024x2)
{
    EXPECT_TRUE(ExecuteTest(1, 1, 1024, 2, 4096, 2, W_X_I) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_8x1x4096_8x4096x1024)
{
    EXPECT_TRUE(ExecuteTest(1, 8, 1, 4096, 1, 1024) == 0);
}
#endif
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_1x1024_1024x4096_reordering_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 1, 1, 1024, 1, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1024_1024x4096_reordering_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 1, 2, 1024, 2, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1x1024_2x1024x4096_reordering_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 2, 1, 1024, 1, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x2x1024_2x1024x4096_reordering_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 2, 2, 1024, 2, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x256_64x256x64_reordering_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 64, 1, 256, 1, 64, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x1024_64x1024x64_reordering_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 64, 1, 1024, 1, 64, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_4x1x4096_4x4096x1024_reordering_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 4, 1, 4096, 1, 1024, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_1x1024_1024x4096_reordering_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 1, 1, 1024, 1, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1024_1024x4096_reordering_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 1, 2, 1024, 2, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1x1024_2x1024x4096_reordering_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 2, 1, 1024, 1, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x2x1024_2x1024x4096_reordering_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 2, 2, 1024, 2, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x256_64x256x64_reordering_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 64, 1, 256, 1, 64, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x1024_64x1024x64_reordering_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 64, 1, 1024, 1, 64, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_4x1x4096_4x4096x1024_reordering_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 4, 1, 4096, 1, 1024, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_1x1024_1024x4096_reordering_on_device_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 1, 1, 1024, 1, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1024_1024x4096_reordering_on_device_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 1, 2, 1024, 2, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1x1024_2x1024x4096_reordering_on_device_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 2, 1, 1024, 1, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x2x1024_2x1024x4096_reordering_on_device_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 2, 2, 1024, 2, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x256_64x256x64_reordering_on_device_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 64, 1, 256, 1, 64, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x1024_64x1024x64_reordering_on_device_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 64, 1, 1024, 1, 64, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_4x1x4096_4x4096x1024_reordering_on_device_from_device)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 4, 1, 4096, 1, 1024, true) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_1x1024_1024x4096_reordering_on_device_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 1, 1, 1024, 1, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1024_1024x4096_reordering_on_device_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 1, 2, 1024, 2, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x1x1024_2x1024x4096_reordering_on_device_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 2, 1, 1024, 1, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_2x2x1024_2x1024x4096_reordering_on_device_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 2, 2, 1024, 2, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x256_64x256x64_reordering_on_device_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 64, 1, 256, 1, 64, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x1024_64x1024x64_reordering_on_device_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 64, 1, 1024, 1, 64, false) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_4x1x4096_4x4096x1024_reordering_on_device_from_host)
{
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 4, 1, 4096, 1, 1024, false) == 0);
}
