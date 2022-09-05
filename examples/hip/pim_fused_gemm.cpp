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

class PimFusedGemmTest
{
   public:
    PimFusedGemmTest(unsigned n_, unsigned c_, unsigned h_, unsigned in_w0_, unsigned out_w0_, unsigned out_w1_,
                     PimActFunc act0_, bool has_bias0_, PimActFunc act1_, bool has_bias1_)
        : n(n_),
          c(c_),
          h(h_),
          in_w0(in_w0_),
          out_w0(out_w0_),
          out_w1(out_w1_),
          act0(act0_),
          has_bias0(has_bias0_),
          act1(act1_),
          has_bias1(has_bias1_)
    {
        if (!is_support_activation(act0) || !is_support_activation(act1)) {
            throw invalid_argument("Invalid activation type");
        }

        in_w1 = out_w0;  // in_w1 must be equal to out_w0

        in_size0 = n * c * h * in_w0;
        wgt_size0 = n * c * in_w0 * out_w0;
        out_size0 = n * c * h * out_w0;
        in_size1 = n * c * h * in_w1;
        wgt_size1 = n * c * in_w1 * out_w1;
        out_size1 = n * c * h * out_w1;

        desc0 = PimCreateGemmDesc(n, c, h, in_w0, out_w0, PIM_FP16);
        h_i0 = PimCreateBo(desc0, MEM_TYPE_HOST, GEMM_INPUT);
        h_w0 = PimCreateBo(desc0, MEM_TYPE_HOST, GEMM_WEIGHT);
        h_b0 = PimCreateBo(desc0, MEM_TYPE_HOST, GEMM_BIAS);
        h_o0 = PimCreateBo(desc0, MEM_TYPE_HOST, GEMM_OUTPUT);
        d_i0 = PimCreateBo(desc0, MEM_TYPE_DEVICE, GEMM_INPUT);
        d_w0 = PimCreateBo(desc0, MEM_TYPE_DEVICE, GEMM_WEIGHT);
        d_b0 = PimCreateBo(desc0, MEM_TYPE_DEVICE, GEMM_BIAS);
        d_o0 = PimCreateBo(desc0, MEM_TYPE_DEVICE, GEMM_OUTPUT);

        desc1 = PimCreateGemmDesc(n, c, h, in_w1, out_w1, PIM_FP16);
        h_i1 = PimCreateBo(desc1, MEM_TYPE_HOST, GEMM_INPUT);
        h_w1 = PimCreateBo(desc1, MEM_TYPE_HOST, GEMM_WEIGHT);
        h_b1 = PimCreateBo(desc1, MEM_TYPE_HOST, GEMM_BIAS);
        h_o1 = PimCreateBo(desc1, MEM_TYPE_HOST, GEMM_OUTPUT);
        d_i1 = PimCreateBo(desc1, MEM_TYPE_DEVICE, GEMM_INPUT);
        d_w1 = PimCreateBo(desc1, MEM_TYPE_DEVICE, GEMM_WEIGHT);
        d_b1 = PimCreateBo(desc1, MEM_TYPE_DEVICE, GEMM_BIAS);
        d_o1 = PimCreateBo(desc1, MEM_TYPE_DEVICE, GEMM_OUTPUT);

        golden = PimCreateBo(desc1, MEM_TYPE_HOST, GEMM_OUTPUT);
    }

    PimFusedGemmTest(unsigned n_, unsigned c_, unsigned h_, unsigned in_w0_, unsigned out_w0_, unsigned out_w1_)
        : PimFusedGemmTest(n_, c_, h_, in_w0_, out_w0_, out_w1_, ACT_RELU, true, ACT_RELU, true)
    {
    }

    ~PimFusedGemmTest()
    {
        PimDestroyBo(h_i0);
        PimDestroyBo(h_w0);
        PimDestroyBo(h_b0);
        PimDestroyBo(h_o0);
        PimDestroyBo(d_i0);
        PimDestroyBo(d_w0);
        PimDestroyBo(d_b0);
        PimDestroyBo(d_o0);

        PimDestroyBo(h_i1);
        PimDestroyBo(h_w1);
        PimDestroyBo(h_b1);
        PimDestroyBo(h_o1);
        PimDestroyBo(d_i1);
        PimDestroyBo(d_w1);
        PimDestroyBo(d_b1);
        PimDestroyBo(d_o1);

        PimDestroyBo(golden);
    }

    void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.2f)
    {
        set_rand_half_data((half*)h_i0->data, half(variation), in_size0);
        set_rand_half_data((half*)h_w0->data, half(variation), wgt_size0);
        set_rand_half_data((half*)h_b0->data, half(variation), out_size0);
        set_half_data((half*)h_o0->data, half(0.0), out_size0);

        set_rand_half_data((half*)h_w1->data, half(variation), wgt_size1);
        set_rand_half_data((half*)h_b1->data, half(variation), out_size1);
        set_half_data((half*)h_o1->data, half(0.0), out_size1);
        set_half_data((half*)golden->data, half(0.0), out_size1);

        half* in_data0 = (half*)h_i0->data;
        half* wgt_data0 = (half*)h_w0->data;
        half* out_data0 = (half*)h_o0->data;
        half* wgt_data1 = (half*)h_w1->data;
        half* out_data1 = (half*)golden->data;

        // first gemm
        for (int nc_i = 0; nc_i < n * c; nc_i++) {
            matmulCPU(in_data0, wgt_data0, out_data0, h, out_w0, in_w0, half(alpha), half(beta));
            in_data0 += (in_w0 * h);
            wgt_data0 += (in_w0 * out_w0);
            out_data0 += (out_w0 * h);
        }
        if (has_bias0) {
            addBiasCPU((half*)h_o0->data, (half*)h_b0->data, out_size0);
        }
        if (act0 == ACT_RELU) {
            reluCPU((half*)h_o0->data, out_size0);
        }

        // second gemm
        half* in_data1 = (half*)h_o0->data;
        for (int nc_i = 0; nc_i < n * c; nc_i++) {
            matmulCPU(in_data1, wgt_data1, out_data1, h, out_w1, in_w1, half(alpha), half(beta));
            in_data1 += (in_w1 * h);
            wgt_data1 += (in_w1 * out_w1);
            out_data1 += (out_w1 * h);
        }
        if (has_bias1) {
            addBiasCPU((half*)golden->data, (half*)h_b1->data, out_size1);
        }
        if (act1 == ACT_RELU) {
            reluCPU((half*)golden->data, out_size1);
        }

        PimCopyMemory(d_i0, h_i0, HOST_TO_DEVICE);
        PimCopyMemory(d_w0, h_w0, HOST_TO_DEVICE);
        if (has_bias0) {
            PimCopyMemory(d_b0, h_b0, HOST_TO_DEVICE);
        } else {
            d_b0 = nullptr;
        }

        PimCopyMemory(d_w1, h_w1, HOST_TO_DEVICE);
        PimCopyMemory(d_b1, h_b1, HOST_TO_DEVICE);
        if (has_bias1) {
            PimCopyMemory(d_b1, h_b1, HOST_TO_DEVICE);
        } else {
            d_b1 = nullptr;
        }
    }

    void run(bool block = true, unsigned niter = 1)
    {
        for (unsigned i = 0; i < niter; ++i) {
            (void)PimExecuteGemm(d_o0, d_i0, d_w0, d_b0, act0, nullptr, false);
            (void)PimExecuteGemm(d_o1, d_o0, d_w1, d_b1, act1, nullptr, block);
            if (!block) PimSynchronize();
        }
        PimCopyMemory(h_o1, d_o1, DEVICE_TO_HOST);
    }

    int validate(float epsilon = 1.0f)
    {
        return compare_half_relative((half*)golden->data, (half*)h_o1->data, out_size1, epsilon);
    }

   private:
    bool is_support_activation(const PimActFunc& act) { return (act == ACT_RELU || act == NONE) ? true : false; }

    // First  (n, c, h, in_w0) * (n, c, in_w0, out_w0) = (n, c, h, out_w0)
    // Second (n, c, h, out_w0) * (n, c, out_w0, out_w1) = (n, c, h, out_w1)
    unsigned n;
    unsigned c;
    unsigned h;
    unsigned in_w0;
    unsigned out_w0, out_w1;
    unsigned in_w1;  // must be same to out_w0

    PimActFunc act0;
    bool has_bias0;
    PimActFunc act1;
    bool has_bias1;

    unsigned in_size0, in_size1;
    unsigned wgt_size0, wgt_size1;
    unsigned out_size0, out_size1;

    PimGemmDesc *desc0, *desc1;
    PimBo *h_i0, *h_w0, *h_b0, *h_o0;  // input, weight, bias, output
    PimBo *h_i1, *h_w1, *h_b1, *h_o1;  // input, weight, bias, output
    PimBo *d_i0, *d_w0, *d_b0, *d_o0;
    PimBo *d_i1, *d_w1, *d_b1, *d_o1;
    PimBo* golden;
};

class PimFusedGemmTestFixture : public ::testing::Test
{
   protected:
    virtual void SetUp() override
    {
        PimInitialize(RT_TYPE_HIP, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown() override { PimDeinitialize(); }

    int ExecuteTest(unsigned n, unsigned c, unsigned h, unsigned in_w0, unsigned out_w0, unsigned out_w1,
                    bool has_bias0 = true, PimActFunc act0 = ACT_RELU, bool has_bias1 = true, PimActFunc act1 = NONE,
                    bool block = true)
    {
        PimFusedGemmTest pimFusedGemmTest =
            PimFusedGemmTest(n, c, h, in_w0, out_w0, out_w1, act0, has_bias0, act1, has_bias1);
        pimFusedGemmTest.prepare();
        pimFusedGemmTest.run(block);
        return pimFusedGemmTest.validate();
    }
};

TEST_F(PimFusedGemmTestFixture, pim_fused_gemm_bias_relu_4x1x1024_4x1024x4096_4x4096x1024)
{
    EXPECT_TRUE(ExecuteTest(1, 4, 1, 1024, 4096, 1024, true, ACT_RELU, true, NONE) == 0);
}
TEST_F(PimFusedGemmTestFixture, pim_fused_gemm_bias_relu_8x1x1024_8x1024x4096_8x4096x1024)
{
    EXPECT_TRUE(ExecuteTest(1, 8, 1, 1024, 4096, 1024, true, ACT_RELU, true, NONE) == 0);
}
