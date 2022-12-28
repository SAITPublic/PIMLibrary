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
#include "test_util.h"

using half_float::half;
using namespace std;

class PimFusedGemmTest : public Testing
{
   public:
    PimFusedGemmTest(unsigned n, unsigned c, unsigned h, unsigned in_w0, unsigned out_w0, unsigned out_w1,
                     PimActFunc act0, bool has_bias0, PimActFunc act1, bool has_bias1)
        : n_(n),
          c_(c),
          h_(h),
          in_w0_(in_w0),
          out_w0_(out_w0),
          out_w1_(out_w1),
          act0_(act0),
          has_bias0_(has_bias0),
          act1_(act1),
          has_bias1_(has_bias1)
    {
        if (!is_support_activation(act0_) || !is_support_activation(act1_)) {
            throw invalid_argument("Invalid activation type");
        }

        in_w1_ = out_w0_;  // in_w1_ must be equal to out_w0_

        in_size0_ = n_ * c_ * h_ * in_w0_;
        wgt_size0_ = n_ * c_ * in_w0_ * out_w0_;
        out_size0_ = n_ * c_ * h_ * out_w0_;
        in_size1_ = n_ * c_ * h_ * in_w1_;
        wgt_size1_ = n_ * c_ * in_w1_ * out_w1_;
        out_size1_ = n_ * c_ * h_ * out_w1_;

        desc0_ = PimCreateGemmDesc(n_, c_, h_, in_w0_, h_, out_w0_, PIM_FP16);
        h_i0_ = PimCreateBo(desc0_, MEM_TYPE_HOST, GEMM_INPUT);
        h_w0_ = PimCreateBo(desc0_, MEM_TYPE_HOST, GEMM_WEIGHT);
        h_b0_ = PimCreateBo(desc0_, MEM_TYPE_HOST, GEMM_BIAS);
        h_o0_ = PimCreateBo(desc0_, MEM_TYPE_HOST, GEMM_OUTPUT);
        d_i0_ = PimCreateBo(desc0_, MEM_TYPE_DEVICE, GEMM_INPUT);
        d_w0_ = PimCreateBo(desc0_, MEM_TYPE_DEVICE, GEMM_WEIGHT);
        d_b0_ = PimCreateBo(desc0_, MEM_TYPE_DEVICE, GEMM_BIAS);
        d_o0_ = PimCreateBo(desc0_, MEM_TYPE_DEVICE, GEMM_OUTPUT);

        desc1_ = PimCreateGemmDesc(n_, c_, h_, in_w1_, h_, out_w1_, PIM_FP16);
        h_i1_ = PimCreateBo(desc1_, MEM_TYPE_HOST, GEMM_INPUT);
        h_w1_ = PimCreateBo(desc1_, MEM_TYPE_HOST, GEMM_WEIGHT);
        h_b1_ = PimCreateBo(desc1_, MEM_TYPE_HOST, GEMM_BIAS);
        h_o1_ = PimCreateBo(desc1_, MEM_TYPE_HOST, GEMM_OUTPUT);
        d_i1_ = PimCreateBo(desc1_, MEM_TYPE_DEVICE, GEMM_INPUT);
        d_w1_ = PimCreateBo(desc1_, MEM_TYPE_DEVICE, GEMM_WEIGHT);
        d_b1_ = PimCreateBo(desc1_, MEM_TYPE_DEVICE, GEMM_BIAS);
        d_o1_ = PimCreateBo(desc1_, MEM_TYPE_DEVICE, GEMM_OUTPUT);

        golden_ = PimCreateBo(desc1_, MEM_TYPE_HOST, GEMM_OUTPUT);
    }

    PimFusedGemmTest(unsigned n, unsigned c, unsigned h, unsigned in_w0, unsigned out_w0, unsigned out_w1)
        : PimFusedGemmTest(n, c, h, in_w0, out_w0, out_w1, ACT_RELU, true, ACT_RELU, true)
    {
    }

    ~PimFusedGemmTest(void)
    {
        PimDestroyBo(h_i0_);
        PimDestroyBo(h_w0_);
        PimDestroyBo(h_b0_);
        PimDestroyBo(h_o0_);
        PimDestroyBo(d_i0_);
        PimDestroyBo(d_w0_);
        PimDestroyBo(d_b0_);
        PimDestroyBo(d_o0_);

        PimDestroyBo(h_i1_);
        PimDestroyBo(h_w1_);
        PimDestroyBo(h_b1_);
        PimDestroyBo(h_o1_);
        PimDestroyBo(d_i1_);
        PimDestroyBo(d_w1_);
        PimDestroyBo(d_b1_);
        PimDestroyBo(d_o1_);

        PimDestroyBo(golden_);
    }

    virtual void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.2f) override
    {
        set_rand_half_data((half*)h_i0_->data, half(variation), in_size0_);
        set_rand_half_data((half*)h_w0_->data, half(variation), wgt_size0_);
        set_rand_half_data((half*)h_b0_->data, half(variation), out_size0_);
        set_half_data((half*)h_o0_->data, half(0.0), out_size0_);

        set_rand_half_data((half*)h_w1_->data, half(variation), wgt_size1_);
        set_rand_half_data((half*)h_b1_->data, half(variation), out_size1_);
        set_half_data((half*)h_o1_->data, half(0.0), out_size1_);
        set_half_data((half*)golden_->data, half(0.0), out_size1_);

        half* in_data0 = (half*)h_i0_->data;
        half* wgt_data0 = (half*)h_w0_->data;
        half* out_data0 = (half*)h_o0_->data;
        half* wgt_data1 = (half*)h_w1_->data;
        half* out_data1 = (half*)golden_->data;

        // first gemm
        for (int nc_i = 0; nc_i < n_ * c_; nc_i++) {
            matmulCPU(in_data0, wgt_data0, out_data0, h_, out_w0_, in_w0_, half(alpha), half(beta));
            in_data0 += (in_w0_ * h_);
            wgt_data0 += (in_w0_ * out_w0_);
            out_data0 += (out_w0_ * h_);
        }
        if (has_bias0_) {
            addBiasCPU((half*)h_o0_->data, (half*)h_b0_->data, out_size0_);
        }
        if (act0_ == ACT_RELU) {
            reluCPU((half*)h_o0_->data, out_size0_);
        }

        // second gemm
        half* in_data1 = (half*)h_o0_->data;
        for (int nc_i = 0; nc_i < n_ * c_; nc_i++) {
            matmulCPU(in_data1, wgt_data1, out_data1, h_, out_w1_, in_w1_, half(alpha), half(beta));
            in_data1 += (in_w1_ * h_);
            wgt_data1 += (in_w1_ * out_w1_);
            out_data1 += (out_w1_ * h_);
        }
        if (has_bias1_) {
            addBiasCPU((half*)golden_->data, (half*)h_b1_->data, out_size1_);
        }
        if (act1_ == ACT_RELU) {
            reluCPU((half*)golden_->data, out_size1_);
        }

        PimCopyMemory(d_i0_, h_i0_, HOST_TO_DEVICE);
        PimCopyMemory(d_w0_, h_w0_, HOST_TO_DEVICE);
        if (has_bias0_) {
            PimCopyMemory(d_b0_, h_b0_, HOST_TO_DEVICE);
        } else {
            d_b0_ = nullptr;
        }

        PimCopyMemory(d_w1_, h_w1_, HOST_TO_DEVICE);
        PimCopyMemory(d_b1_, h_b1_, HOST_TO_DEVICE);
        if (has_bias1_) {
            PimCopyMemory(d_b1_, h_b1_, HOST_TO_DEVICE);
        } else {
            d_b1_ = nullptr;
        }
    }

    virtual void run(bool block = true, unsigned niter = 1) override
    {
        for (unsigned i = 0; i < niter; ++i) {
            (void)PimExecuteGemm(d_o0_, d_i0_, d_w0_, d_b0_, act0_, I_X_W, nullptr, false);
            (void)PimExecuteGemm(d_o1_, d_o0_, d_w1_, d_b1_, act1_, I_X_W, nullptr, block);
            if (!block) PimSynchronize();
        }
        PimCopyMemory(h_o1_, d_o1_, DEVICE_TO_HOST);
    }

    virtual int validate(float epsilon = 0.1f) override
    {
        return compare_half_relative((half*)golden_->data, (half*)h_o1_->data, out_size1_, epsilon);
    }

   private:
    bool is_support_activation(const PimActFunc& act) { return (act == ACT_RELU || act == NONE) ? true : false; }
    // First  (n, c, h, in_w0) * (n, c, in_w0, out_w0) = (n, c, h, out_w0)
    // Second (n, c, h, out_w0) * (n, c, out_w0, out_w1) = (n, c, h, out_w1)
    unsigned n_;
    unsigned c_;
    unsigned h_;
    unsigned in_w0_;
    unsigned out_w0_, out_w1_;
    unsigned in_w1_;  // must be same to out_w0

    PimActFunc act0_;
    bool has_bias0_;
    PimActFunc act1_;
    bool has_bias1_;

    unsigned in_size0_, in_size1_;
    unsigned wgt_size0_, wgt_size1_;
    unsigned out_size0_, out_size1_;

    PimGemmDesc *desc0_, *desc1_;
    PimBo *h_i0_, *h_w0_, *h_b0_, *h_o0_;  // input, weight, bias, output
    PimBo *h_i1_, *h_w1_, *h_b1_, *h_o1_;  // input, weight, bias, output
    PimBo *d_i0_, *d_w0_, *d_b0_, *d_o0_;
    PimBo *d_i1_, *d_w1_, *d_b1_, *d_o1_;
    PimBo* golden_;
};

class PimFusedGemmTestFixture : public ::testing::Test
{
   protected:
    void SetUp(PimRuntimeType plt)
    {
        PimInitialize(plt, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown(void) override { PimDeinitialize(); }
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
