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

class PimGemmTest
{
   public:
    PimGemmTest(unsigned n_, unsigned c_, unsigned h_, unsigned in_w_, unsigned out_w_, PimActFunc act_, bool has_bias_)
        : n(n_), c(c_), h(h_), in_w(in_w_), out_w(out_w_), act(act_), has_bias(has_bias_)
    {
        if (!is_support_activation(act)) {
            throw invalid_argument("Invalid activation type");
        }

        in_size = n * c * h * in_w;
        wgt_size = n * c * in_w * out_w;
        out_size = n * c * h * out_w;

        desc = PimCreateGemmDesc(n, c, h, in_w, out_w, PIM_FP16);
        h_i = PimCreateBo(desc, MEM_TYPE_HOST, GEMM_INPUT);
        h_w = PimCreateBo(desc, MEM_TYPE_HOST, GEMM_WEIGHT);
        h_b = PimCreateBo(desc, MEM_TYPE_HOST, GEMM_BIAS);
        h_o = PimCreateBo(desc, MEM_TYPE_HOST, GEMM_OUTPUT);
        d_i = PimCreateBo(desc, MEM_TYPE_DEVICE, GEMM_INPUT);
        d_w = PimCreateBo(desc, MEM_TYPE_DEVICE, GEMM_WEIGHT);
        d_b = PimCreateBo(desc, MEM_TYPE_DEVICE, GEMM_BIAS);
        d_o = PimCreateBo(desc, MEM_TYPE_DEVICE, GEMM_OUTPUT);
        golden = PimCreateBo(desc, MEM_TYPE_HOST, GEMM_OUTPUT);
    }

    PimGemmTest(unsigned n_, unsigned c_, unsigned h_, unsigned in_w_, unsigned out_w_)
        : PimGemmTest(n_, c_, h_, in_w_, out_w_, ACT_RELU, true)
    {
    }

    ~PimGemmTest()
    {
        PimDestroyBo(h_i);
        PimDestroyBo(h_w);
        PimDestroyBo(h_b);
        PimDestroyBo(h_o);
        PimDestroyBo(golden);
        PimDestroyBo(d_i);
        PimDestroyBo(d_w);
        PimDestroyBo(d_b);
        PimDestroyBo(d_o);
    }

    void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.2f)
    {
        set_half_data((half*)golden->data, half(0.0), out_size);
        set_half_data((half*)h_o->data, half(0.0), out_size);
        set_rand_half_data((half*)h_i->data, half(variation), in_size);
        set_rand_half_data((half*)h_w->data, half(variation), wgt_size);
        set_rand_half_data((half*)h_b->data, half(variation), out_size);

        half* h_i_data = (half*)h_i->data;
        half* h_w_data = (half*)h_w->data;
        half* golden_data = (half*)golden->data;
        for (int nc_i = 0; nc_i < n * c; nc_i++) {
            matmulCPU(h_i_data, h_w_data, golden_data, h, out_w, in_w, half(alpha), half(beta));
            h_i_data += (in_w * h);
            h_w_data += (in_w * out_w);
            golden_data += (out_w * h);
        }
        if (has_bias) {
            addBiasCPU((half*)golden->data, (half*)h_b->data, out_size);
        }
        if (act == ACT_RELU) {
            reluCPU((half*)golden->data, out_size);
        }
        PimCopyMemory(d_i, h_i, HOST_TO_DEVICE);
        PimCopyMemory(d_w, h_w, HOST_TO_DEVICE);
        PimCopyMemory(d_o, h_o, HOST_TO_DEVICE);

        if (has_bias) {
            PimCopyMemory(d_b, h_b, HOST_TO_DEVICE);
        } else {
            d_b = nullptr;
        }
    }

    void run(bool block = true, unsigned niter = 1)
    {
        for (unsigned i = 0; i < niter; ++i) {
            (void)PimExecuteGemm(d_o, d_i, d_w, d_b, act, nullptr, block);
            if (!block) PimSynchronize();
        }
        PimCopyMemory(h_o, d_o, DEVICE_TO_HOST);
    }

    int validate(float epsilon = 1.0f)
    {
        return compare_half_relative((half*)golden->data, (half*)h_o->data, out_size, epsilon);
    }

   private:
    bool is_support_activation(const PimActFunc& act) { return (act == ACT_RELU || act == NONE) ? true : false; }

    // (n, c, h, in_w) * (n, c, in_w, out_w) = (n, c, h, out_w)
    unsigned n;
    unsigned c;
    unsigned h;
    unsigned in_w;
    unsigned out_w;

    PimActFunc act;
    bool has_bias;

    unsigned in_size;
    unsigned wgt_size;
    unsigned out_size;

    PimGemmDesc* desc;
    PimBo *h_i, *h_w, *h_b, *h_o;  // input, weight, bias, output
    PimBo *d_i, *d_w, *d_b, *d_o;
    PimBo* golden;
};

class PimGemmTestFixture : public ::testing::Test
{
   protected:
    virtual void SetUp() override
    {
        PimInitialize(RT_TYPE_HIP, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown() override { PimDeinitialize(); }

    int ExecuteTest(unsigned n, unsigned c, unsigned h, unsigned in_w, unsigned out_w, bool has_bias = true,
                    bool block = true, PimActFunc act = ACT_RELU)
    {
        PimGemmTest pimGemmTest = PimGemmTest(n, c, h, in_w, out_w, act, has_bias);
        pimGemmTest.prepare();
        pimGemmTest.run(block);
        return pimGemmTest.validate();
    }
};

TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_1x1024_1024x4096) { EXPECT_TRUE(ExecuteTest(1, 1, 1, 1024, 4096) == 0); }
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_8x1024_1024x4096) { EXPECT_TRUE(ExecuteTest(1, 1, 8, 1024, 4096) == 0); }
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_4x1x1024_4x1024x4096)
{
    EXPECT_TRUE(ExecuteTest(1, 4, 1, 1024, 4096) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_4x8x1024_4x1024x4096)
{
    EXPECT_TRUE(ExecuteTest(1, 4, 8, 1024, 4096) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x256_64x256x64) { EXPECT_TRUE(ExecuteTest(1, 64, 1, 256, 64) == 0); }
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_64x1x1024_64x1024x64)
{
    EXPECT_TRUE(ExecuteTest(1, 64, 1, 1024, 64) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_4x1x4096_4x4096x1024)
{
    EXPECT_TRUE(ExecuteTest(1, 4, 1, 4096, 1024) == 0);
}
TEST_F(PimGemmTestFixture, pim_gemm_bias_relu_8x1x4096_8x4096x1024)
{
    EXPECT_TRUE(ExecuteTest(1, 8, 1, 4096, 1024) == 0);
}
