/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */
#include "elt_perf.h"

using half_float::half;
using namespace std;

PimEltTest::PimEltTest(unsigned n, unsigned c, unsigned in_h, unsigned in_w, PimPrecision precision)
    : n_(n), c_(c), in_h_(in_h), in_w_(in_w), out_h_(in_h), out_w_(in_w), precision_(precision)
{
    in_size_ = n_ * c_ * in_h_ * in_w_;
    out_size_ = n_ * c_ * out_h_ * out_w_;
    flt_ops_ = out_size_;
    desc_ = PimCreateDesc(n_, c_, in_h_, in_w_, precision_);
    h_i_1_ = PimCreateBo(desc_, MEM_TYPE_HOST, ELT_OP);
    h_i_2_ = PimCreateBo(desc_, MEM_TYPE_HOST, ELT_OP);
    h_o_ = PimCreateBo(desc_, MEM_TYPE_HOST, ELT_OP);
    d_i_1_ = PimCreateBo(desc_, MEM_TYPE_PIM, ELT_OP);
    d_i_2_ = PimCreateBo(desc_, MEM_TYPE_PIM, ELT_OP);
    d_o_ = PimCreateBo(desc_, MEM_TYPE_PIM, ELT_OP);
    golden_ = PimCreateBo(desc_, MEM_TYPE_HOST, ELT_OP);
}

PimEltTest::~PimEltTest()
{
    PimDestroyBo(h_i_1_);
    PimDestroyBo(h_i_2_);
    PimDestroyBo(h_o_);
    PimDestroyBo(d_i_1_);
    PimDestroyBo(d_i_2_);
    PimDestroyBo(d_o_);
    PimDestroyBo(golden_);
}

void PimEltTest::prepare(float variation)
{
    set_rand_half_data((half_float::half*)h_i_1_->data, (half_float::half)0.5, in_size_);
    set_rand_half_data((half_float::half*)h_i_2_->data, (half_float::half)0.5, in_size_);
    addCPU((half_float::half*)h_i_1_->data, (half_float::half*)h_i_2_->data, (half_float::half*)golden_->data,
           in_size_);
    PimCopyMemory(d_i_1_, h_i_1_, HOST_TO_PIM);
    PimCopyMemory(d_i_2_, h_i_2_, HOST_TO_PIM);
}

void PimEltTest::execute_op(bool block)
{
    PimExecuteAdd(d_o_, d_i_1_, d_i_2_, nullptr, true);
    if (!block) PimSynchronize();
}

void PimEltTest::finalize() { PimCopyMemory(h_o_, d_o_, PIM_TO_HOST); }
int PimEltTest::validate(float epsilon)
{
    return compare_half_relative((half*)h_o_->data, (half*)golden_->data, out_size_, epsilon);
}

double PimEltTest::get_flt_ops() { return flt_ops_; }
PimReluTest::PimReluTest(unsigned n, unsigned c, unsigned in_h, unsigned in_w, PimPrecision precision)
    : n_(n), c_(c), in_h_(in_h), in_w_(in_w), out_h_(in_h), out_w_(in_w), precision_(precision)
{
    in_size_ = n_ * c_ * in_h_ * in_w_;
    out_size_ = n_ * c_ * out_h_ * out_w_;
    flt_ops_ = out_size_;
    desc_ = PimCreateDesc(n_, c_, in_h_, in_w_, precision_, OP_RELU);
    h_i_ = PimCreateBo(desc_, MEM_TYPE_HOST);
    h_o_ = PimCreateBo(desc_, MEM_TYPE_HOST);
    d_i_ = PimCreateBo(desc_, MEM_TYPE_PIM);
    d_o_ = PimCreateBo(desc_, MEM_TYPE_PIM);
    golden_ = PimCreateBo(desc_, MEM_TYPE_HOST);
}

PimReluTest::~PimReluTest()
{
    PimDestroyBo(h_i_);
    PimDestroyBo(h_o_);
    PimDestroyBo(d_i_);
    PimDestroyBo(d_o_);
    PimDestroyBo(golden_);
}

void PimReluTest::calculate_relu_cpu(half_float::half* input, half_float::half* output, int input_len)
{
    for (int i = 0; i < input_len; i++) {
        output[i] = input[i];
        if (input[i] < 0) {
            output[i] = 0;
        }
    }
}

void PimReluTest::prepare(float variation)
{
    set_rand_half_data((half_float::half*)h_i_->data, (half_float::half)0.5, in_size_);
    calculate_relu_cpu((half_float::half*)h_i_->data, (half_float::half*)golden_->data, in_size_);
    PimCopyMemory(d_i_, h_i_, HOST_TO_PIM);
}

void PimReluTest::execute_op(bool block)
{
    PimExecuteRelu(d_o_, d_i_, nullptr, block);
    if (!block) PimSynchronize();
}

void PimReluTest::finalize() { PimCopyMemory(h_o_, d_o_, PIM_TO_HOST); }
int PimReluTest::validate(float epsilon)
{
    return compare_half_relative((half*)h_o_->data, (half*)golden_->data, out_size_, epsilon);
}

double PimReluTest::get_flt_ops() { return flt_ops_; }
int PimEltTestFixture::ExecuteTest()
{
    PimEltTest pimEltTest = PimEltTest(num_batch, num_channels, input_height, input_width, precision);
    pimEltTest.prepare();
    for (int i = 0; i < num_iter; i++) {
        // considering first iteration as warm up iteration.
        if (warmup) {
            pimEltTest.execute_op(true);
            warmup = false;
        } else {
            pimEltTest.execute_op(block);
            Tock();
            avg_kernel_time += calculate_elapsed_time();
            std::cout << "Time taken for iter " << i << " : " << time_duration.count() << std::endl;
        }
    }
    pimEltTest.finalize();
    kernel_execution_time = avg_kernel_time / (double)(num_iter - 1);
    calculate_gflops(pimEltTest.get_flt_ops());
    return pimEltTest.validate();
}

int PimReluTestFixture::ExecuteTest()
{
    PimReluTest pimReluTest = PimReluTest(num_batch, num_channels, input_height, input_width, precision);
    pimReluTest.prepare();
    for (int i = 0; i < num_iter; i++) {
        // considering first iteration as warm up iteration.
        if (warmup) {
            pimReluTest.execute_op(true);
            warmup = false;
        } else {
            pimReluTest.execute_op(block);
            Tock();
            avg_kernel_time += calculate_elapsed_time();
            std::cout << "Time taken for iter " << i << " : " << time_duration.count() << std::endl;
        }
    }
    pimReluTest.finalize();
    kernel_execution_time = avg_kernel_time / (double)(num_iter - 1);
    calculate_gflops(pimReluTest.get_flt_ops());
    return pimReluTest.validate();
    return 1;
}
