/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */
#include "gemm_perf.h"

using half_float::half;
using namespace std;

PimGemmTest::PimGemmTest(unsigned n, unsigned c, unsigned in_h, unsigned in_w, unsigned out_h, unsigned out_w,
                         PimActFunc act, bool has_bias, PimGemmOrder gemm_order)
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
    flt_ops_ = out_h_ * out_w_;
    if (gemm_order_ == W_X_I) {
        wgt_size_ = n_ * c_ * in_h_ * out_h_;
        flt_ops_ *= (2 * in_h_ - 1);
    } else {
        wgt_size_ = n_ * c_ * in_w_ * out_w_;
        flt_ops_ *= (2 * in_w_ - 1);
    }

    desc_ = PimCreateGemmDesc(n_, c_, in_h_, in_w_, out_h_, out_w_, PIM_FP16, gemm_order);
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

PimGemmTest::~PimGemmTest()
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

void PimGemmTest::prepare(float alpha, float beta, float variation)
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
            matmulCPU(h_w_data, h_i_data, golden_data, out_h_, out_w_, out_w_, half(alpha), half(beta));
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

void PimGemmTest::execute_op(bool block)
{
    (void)PimExecuteGemm(d_o_, d_i_, d_w_, d_b_, act_, gemm_order_, nullptr, block);
    if (!block) PimSynchronize();
}

void PimGemmTest::finalize() { PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST); }
void PimGemmTest::run_with_explicit_reordering(bool use_device_weight, bool block, unsigned niter)
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

int PimGemmTest::validate(float epsilon)
{
    return compare_half_relative((half*)h_o_->data, (half*)golden_->data, out_size_, epsilon);
}

double PimGemmTest::get_flt_ops() { return flt_ops_; }
PimGemmTestFixture::PimGemmTestFixture() {}
int PimGemmTestFixture::ExecuteTest()
{
    act = (parser->get_act_function() == "relu") ? ACT_RELU : NONE;
    has_bias = (parser->get_has_bias()) ? true : false;
    PimGemmTest pimGemmTest = PimGemmTest(num_batch, num_channels, input_height, input_width, output_height,
                                          output_width, act, has_bias, order);
    pimGemmTest.prepare();
    for (int i = 0; i < num_iter; i++) {
        // considering first iteration as warm up iteration.
        if (warmup) {
            pimGemmTest.execute_op(true);
            warmup = false;
        } else {
            Tick();
            pimGemmTest.execute_op(block);
            Tock();
            avg_kernel_time += calculate_elapsed_time();
            std::cout << "Time taken for iter " << i << " : " << time_duration.count() << std::endl;
        }
    }
    pimGemmTest.finalize();
    kernel_execution_time = avg_kernel_time / (double)(num_iter - 1);
    calculate_gflops(pimGemmTest.get_flt_ops());
    return pimGemmTest.validate();
}

int PimGemmTestFixture::ExecuteTestExplicitReordering()
{
    bool use_device_weight = false;
    PimGemmTest pimGemmTest = PimGemmTest(num_batch, num_channels, input_height, input_width, output_height,
                                          output_width, act, has_bias, order);
    pimGemmTest.prepare();
    pimGemmTest.run_with_explicit_reordering(use_device_weight, block);
    return pimGemmTest.validate();
}
