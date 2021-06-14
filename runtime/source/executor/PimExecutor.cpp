/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "executor/PimExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "executor/pim_hip_kernels/pim_op_kernels.pimk"
#include "hip/hip_runtime.h"
#include "utility/pim_dump.hpp"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

extern uint64_t g_pim_base_addr;
namespace pim
{
namespace runtime
{
namespace executor
{
PimExecutor::PimExecutor(PimRuntimeType rt_type, PimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";
    get_pim_block_info(&fbi_);
#ifdef EMULATOR
    pim_emulator_ = pim::runtime::emulator::PimEmulator::get_instance();
    fmtd_size_per_ch_ = 100000;
    max_block_size_ = fbi_.num_pim_chan;
    max_fmtd_size_ = fmtd_size_per_ch_ * max_block_size_;
    is_gemv_tree_ = true;
#endif
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

PimExecutor* PimExecutor::get_instance(PimRuntimeType rt_type, PimPrecision precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " Called";
    static PimExecutor* instance_ = new PimExecutor(rt_type, precision);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return instance_;
}

int PimExecutor::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " Intialization done ";

    int ret = 0;
    int device_id = 0;
    hipGetDeviceProperties(&dev_prop_, 0);
    DLOG(INFO) << " System minor " << dev_prop_.minor << std::endl;
    DLOG(INFO) << " System major " << dev_prop_.major << std::endl;
    DLOG(INFO) << " agent prop name " << dev_prop_.name << std::endl;
    DLOG(INFO) << " device id " << device_id << std::endl;
    DLOG(INFO) << " hip Device prop succeeded " << std::endl;

    pim_manager_ = pim::runtime::manager::PimManager::get_instance(rt_type_, precision_);

    max_crf_size_ = 128;
    int max_srf_size = 2048;

    hipMalloc((void**)&d_srf_bin_buffer_, max_srf_size);
#ifdef EMULATOR
    int reserved_fmtd_size = max_fmtd_size_ * sizeof(PimMemTraceData);
    hipMalloc((void**)&d_fmtd16_, reserved_fmtd_size);
    hipMalloc((void**)&d_fmtd16_size_, sizeof(int));
    hipHostMalloc((void**)&d_emulator_trace_, sizeof(PimMemTracer));

    h_fmtd16_ = (PimMemTraceData*)malloc(reserved_fmtd_size);
    h_fmtd32_ = (PimMemTraceData*)malloc(reserved_fmtd_size);
    h_fmtd16_size_ = (int*)malloc(sizeof(int));
    h_fmtd32_size_ = (int*)malloc(sizeof(int));
#endif
    /* PIM HW can generate only gemv output without reduction sum */
    /* so PimExecutor needs to maintain intermediate output buffer for gemv op */

    pim_manager_->alloc_memory((void**)&pim_gemv_tmp_buffer_, 8*2 * 1024 * 1024, MEM_TYPE_PIM);

    // FIXME: 256 x 1024 size occur error in emulator.
    pim_manager_->alloc_memory((void**)&zero_buffer_, 2 * 1024 * 1024, MEM_TYPE_PIM);
    std::fill_n(zero_buffer_, 256*1024, 0);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    hipFree((void*)d_srf_bin_buffer_);

    for (auto it = crf_lut_.begin(); it != crf_lut_.end(); it++) {
        hipFree((void*)it->second);
    }
    crf_lut_.clear();

#ifdef EMULATOR
    hipFree((void*)d_fmtd16_);
    hipFree((void*)d_fmtd16_size_);
    free(h_fmtd16_);
    free(h_fmtd16_size_);
    free(h_fmtd32_);
    free(h_fmtd32_size_);
#endif
    pim_manager_->free_memory((void*)pim_gemv_tmp_buffer_, MEM_TYPE_PIM);
    pim_manager_->free_memory((void*)zero_buffer_, MEM_TYPE_PIM);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::get_loop_counter(PimOpType op_type, int input_size)
{
    int lc = 0;

    int num_transaction = (input_size / 16) / sizeof(uint16_t);
    int num_parallelism = fbi_.num_pim_blocks * fbi_.num_pim_chan * fbi_.num_pim_rank * fbi_.num_grf;
    int num_tile = num_transaction / num_parallelism;

    if (op_type == OP_GEMV && is_gemv_tree_ == false) {
        lc = input_size / fbi_.trans_size / fbi_.num_grf_A;
    }  else if (op_type == OP_GEMV && is_gemv_tree_ == true) {
        lc = (input_size / fbi_.trans_size / fbi_.num_grf_A / 2) - 1;
    }else {
        lc = num_tile / 2 - 1;
    }

    return lc;
}

int PimExecutor::execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;
    int output_size = output->size;

    uint8_t* crf_bin = find_crf(OP_ELT_ADD, output_size);
    int crf_size = 32;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_ELT_ADD, output_size);
    }

    int num_tile = output_size / (131072 << 1);

    unsigned blocks = 64;
    unsigned threads_per_block = 32;
    hipLaunchKernelGGL(elt_op_pim, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)operand0->data,
                       (uint8_t*)operand1->data, (uint8_t*)g_pim_base_addr, (uint8_t*)output->data, num_tile,
#ifdef EMULATOR
                       (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                       (PimMemTracer*)d_emulator_trace_,
#endif
                       (uint8_t*)crf_bin, crf_size);
#ifdef EMULATOR
    hipStreamSynchronize(stream);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                     OP_ELT_ADD);
    pim_emulator_->execute_elt_op(output, operand0, operand1, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr);
#else
    if (block) hipStreamSynchronize(stream);
#endif

    return ret;
}

int PimExecutor::execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;
    int output_size = output->size;

    uint8_t* crf_bin = find_crf(OP_ELT_MUL, output->size);
    int crf_size = 32;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_ELT_MUL, output->size);
    }

    int num_tile = output_size / (131072 << 1);

    unsigned blocks = 64;
    unsigned threads_per_block = 32;
    hipLaunchKernelGGL(elt_op_pim, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)operand0->data,
                       (uint8_t*)operand1->data, (uint8_t*)g_pim_base_addr, (uint8_t*)output->data, num_tile,
#ifdef EMULATOR
                       (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                       (PimMemTracer*)d_emulator_trace_,
#endif
                       (uint8_t*)crf_bin, crf_size);
#ifdef EMULATOR
    hipStreamSynchronize(stream);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                     OP_ELT_MUL);
    pim_emulator_->execute_elt_op(output, operand0, operand1, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr);
#else
    if (block) hipStreamSynchronize(stream);
#endif
    return ret;
}

int PimExecutor::execute_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int is_gemv_add = 0;

    PimBo* input = operand0;
    PimBo* weight = operand1;
    unsigned blocks = fbi_.num_pim_chan;
    unsigned threads_per_block = 64;

    int memory_size = weight->bshape.w;
    int compute_size = 128 * ceil((float)weight->bshape_r.w / 128);
    if (compute_size < 256) compute_size = 256;
    int out_size = weight->bshape.h;
    int real_out_size = weight->bshape_r.h;
    int n_batch = input->bshape.n;
    int n_compute_tile = compute_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_memory_tile = memory_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_out_tile = out_size / (fbi_.num_pim_chan * fbi_.num_pim_blocks * fbi_.num_grf_B);

    PIM_PROFILE_TICK(CreateCRFBin);

    uint8_t* crf_bin = find_crf(OP_GEMV, compute_size * sizeof(uint16_t));
    int crf_size = 32;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_GEMV, compute_size * sizeof(uint16_t));
    }
    PIM_PROFILE_TOCK(CreateCRFBin);

    PIM_PROFILE_TICK(RunGemvKernel);
    hipLaunchKernelGGL(gemv_pim_64cu_64th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)g_pim_base_addr /* pim control base */, (uint8_t*)weight->data /* pim weight base */,
                       (uint8_t*)pim_gemv_tmp_buffer_, /* pim hw output buffer */
                       (uint8_t*)input->data, (uint8_t*)output->data, n_batch, n_memory_tile, n_compute_tile,
                       n_out_tile, real_out_size,
#ifdef EMULATOR
                       (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                       (PimMemTracer*)d_emulator_trace_,
#endif
                       (uint8_t*)crf_bin, crf_size, is_gemv_add);
#ifndef EMULATOR
    if (block) hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvKernel);
#endif
#ifdef EMULATOR
    hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvKernel);

    PIM_PROFILE_TICK(RunGemvEmulation);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;

    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    pim_emulator_->execute_gemv(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV, g_pim_base_addr,
                                pim_gemv_tmp_buffer_);
    PIM_PROFILE_TOCK(RunGemvEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::execute_gemv_tree(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int is_gemv_add = 0;

    PimBo* input = operand0;
    PimBo* weight = operand1;
    unsigned blocks = fbi_.num_pim_chan;
    unsigned threads_per_block = 64;

    int memory_size = weight->bshape.w;
    int out_size = weight->bshape.h;
    int real_out_size = weight->bshape_r.h;
    int n_batch = input->bshape.n;
    int n_memory_tile = memory_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_out_tile = out_size / (fbi_.num_pim_chan * fbi_.num_pim_blocks * fbi_.num_grf_B);

    PIM_PROFILE_TICK(CreateCRFBin);

    uint8_t* crf_bin = find_crf(OP_GEMV, memory_size * sizeof(uint16_t));
    int crf_size = 64;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_GEMV, memory_size * sizeof(uint16_t));
    }
    PIM_PROFILE_TOCK(CreateCRFBin);

    PIM_PROFILE_TICK(RunGemvKernel);
    hipLaunchKernelGGL(gemv_tree_pim_64cu_64th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)g_pim_base_addr /* pim control base */, (uint8_t*)weight->data /* pim weight base */,
                       (uint8_t*)pim_gemv_tmp_buffer_, /* pim hw output buffer */
                       (uint8_t*)zero_buffer_,
                       (uint8_t*)input->data, (uint8_t*)output->data, n_batch, n_memory_tile, n_memory_tile,
                       n_out_tile, real_out_size,
#ifdef EMULATOR
                       (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                       (PimMemTracer*)d_emulator_trace_,
#endif
                       (uint8_t*)crf_bin, crf_size, is_gemv_add);
#ifndef EMULATOR
    if (block) hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvKernel);
#endif
#ifdef EMULATOR
    hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvKernel);

    PIM_PROFILE_TICK(RunGemvEmulation);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;

    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    pim_emulator_->execute_gemv_tree(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV, g_pim_base_addr,
                                pim_gemv_tmp_buffer_, zero_buffer_);
    PIM_PROFILE_TOCK(RunGemvEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::execute_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int is_gemv_add = 1;

    PimBo* input = operand0;
    PimBo* weight = operand1;
    unsigned blocks = fbi_.num_pim_chan;
    unsigned threads_per_block = 64;

    int memory_size = weight->bshape.w;
    int compute_size = 128 * ceil((float)weight->bshape_r.w / 128);
    if (compute_size < 256) compute_size = 256;
    int out_size = weight->bshape.h;
    int real_out_size = weight->bshape_r.h;
    int n_batch = input->bshape.n;
    int n_compute_tile = compute_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_memory_tile = memory_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_out_tile = out_size / (blocks * fbi_.num_pim_blocks * fbi_.num_grf_B);

    PIM_PROFILE_TICK(CreateCRFBin);
    uint8_t* crf_bin = find_crf(OP_GEMV, compute_size * sizeof(uint16_t));
    int crf_size = 32;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_GEMV, compute_size * sizeof(uint16_t));
    }

    PIM_PROFILE_TOCK(CreateCRFBin);

    PIM_PROFILE_TICK(RunGemvKernel);
    hipLaunchKernelGGL(gemv_pim_64cu_64th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)g_pim_base_addr /* pim control base */, (uint8_t*)weight->data /* pim weight base */,
                       (uint8_t*)pim_gemv_tmp_buffer_, /* pim hw output buffer */
                       (uint8_t*)input->data, (uint8_t*)output->data, n_batch, n_memory_tile, n_compute_tile,
                       n_out_tile, real_out_size,
#ifdef EMULATOR
                       (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                       (PimMemTracer*)d_emulator_trace_,
#endif
                       (uint8_t*)crf_bin, crf_size, is_gemv_add);
#ifndef EMULATOR
    if (block) hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvKernel);
#endif
#ifdef EMULATOR
    hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvKernel);

    PIM_PROFILE_TICK(RunGemvEmulation);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;

    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    pim_emulator_->execute_gemv_add(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV, g_pim_base_addr,
                                    pim_gemv_tmp_buffer_);
    PIM_PROFILE_TOCK(RunGemvEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::execute_relu(PimBo* output, PimBo* pim_data, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    uint8_t* crf_bin = find_crf(OP_RELU, output->size);
    int crf_size = 32;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_RELU, output->size);
    }

    unsigned blocks = fbi_.num_pim_chan;
    unsigned threads_per_block = 32;
    hipLaunchKernelGGL(relu_pim, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)pim_data->data,
                       (uint8_t*)g_pim_base_addr, (uint8_t*)output->data, (int)output->size,
#ifdef EMULATOR
                       (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                       (PimMemTracer*)d_emulator_trace_,
#endif
                       (uint8_t*)crf_bin, crf_size);
#ifdef EMULATOR
    hipStreamSynchronize(stream);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_RELU);
    pim_emulator_->execute_relu(output, pim_data, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr);
#else
    if (block) hipStreamSynchronize(stream);
#endif

    return ret;
}

int PimExecutor::preprocess_srf(PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance, double epsilon,
                                uint8_t* srf_binary)
{
    int num_pim_rank = fbi_.num_pim_rank;
    int num_pim_chan = fbi_.num_pim_chan;

    int cidx = 0;
    int rank = 0;
    int burst_idx = 0;
    int num_stride_reg = 2;
    int num_half_per_reg = 16;

    half* h_srf_binary = reinterpret_cast<half*>(srf_binary);
    half* h_beta = (half*)beta->data;
    half* h_gamma = (half*)gamma->data;
    half* h_var = (half*)variance->data;
    half* h_mean = (half*)mean->data;

    for (int ch_model = 0; ch_model < beta->bshape.c; ch_model++) {
        h_srf_binary[cidx * num_pim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx] =
            1 / sqrt((float)h_var[ch_model] + epsilon);  // scale
        h_srf_binary[cidx * num_pim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 1] =
            h_gamma[ch_model];  // gamma
        h_srf_binary[cidx * num_pim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 8] =
            -(float)h_mean[ch_model] / sqrt((float)h_var[ch_model] + epsilon);  // shift
        h_srf_binary[cidx * num_pim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 9] =
            h_beta[ch_model];  // beta
        rank++;

        if (rank >= num_pim_rank) {
            rank = 0;
            cidx++;
        }
        if (cidx >= num_pim_chan) {
            cidx = 0;
            burst_idx += num_stride_reg;
        }
        if (burst_idx >= 8) {
            std::cout << "error: this is not defined" << std::endl;
        }
    }
    return 0;
}

int PimExecutor::execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                            double epsilon, hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int output_size = output->size;

    uint8_t* crf_bin = find_crf(OP_BN, output_size);
    int crf_size = 64;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_BN, output_size);
    }

    uint8_t* srf_binary = new uint8_t[fbi_.num_pim_chan * fbi_.num_pim_rank * fbi_.trans_size];
    int srf_size = fbi_.num_pim_chan * fbi_.num_pim_rank * fbi_.trans_size;
    preprocess_srf(beta, gamma, mean, variance, epsilon, srf_binary);

    hipMemcpy((void*)d_srf_bin_buffer_, (void*)srf_binary, srf_size, hipMemcpyHostToDevice);

    int num_tile = output_size / (131072 << 1);
    // printf("crf_size:%d, srf_size:%d, output->size:%d\n", crf_size, srf_size, output->size);
    // printf("bshaped(%d,%d,%d,%d)\n", output->bshape.w, output->bshape.h, output->bshape.c, output->bshape.n);
    unsigned blocks = 64;
    unsigned threads_per_block = 32;
    hipLaunchKernelGGL(bn_pim_nr_sip, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)pim_data->data,
                       (uint8_t*)g_pim_base_addr, (uint8_t*)pim_gemv_tmp_buffer_, (uint8_t*)output->data, (int)num_tile,
                       output->bshape.n, output->bshape.c, output->bshape.w,
#ifdef EMULATOR
                       (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                       (PimMemTracer*)d_emulator_trace_,
#endif
                       (uint8_t*)crf_bin, crf_size, (uint8_t*)d_srf_bin_buffer_, srf_size);

#ifdef EMULATOR
    hipStreamSynchronize(stream);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_BN);
    pim_emulator_->execute_bn(output, pim_data, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr, pim_gemv_tmp_buffer_);
#else
    if (block) hipStreamSynchronize(stream);
#endif
    delete[] srf_binary;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::execute_sync(hipStream_t stream) { return hipStreamSynchronize(stream); }
int PimExecutor::execute_dummy(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    hipLaunchKernelGGL(dummy_kernel, dim3(1), dim3(1), 0, 0);
    hipStreamSynchronize(NULL);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

uint8_t* PimExecutor::find_crf(PimOpType op_type, int data_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    uint8_t* addr = nullptr;

    std::map<std::pair<PimOpType, int>, uint8_t*>::const_iterator found =
        crf_lut_.find(std::make_pair(op_type, data_size));
    if (found != crf_lut_.end()) {
        addr = found->second;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return addr;
}

uint8_t* PimExecutor::make_crf_bin(PimOpType op_type, int data_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    uint8_t* h_crf = new uint8_t[max_crf_size_];
    uint8_t* d_crf;
    int crf_size;
    hipMalloc((void**)&d_crf, max_crf_size_);

    int lc = get_loop_counter(op_type, data_size);
    pim_manager_->pim_crf_generator_->gen_binary_with_loop(op_type, lc, h_crf, &crf_size);

    pim_manager_->copy_memory((void*)d_crf, (void*)h_crf, max_crf_size_, HOST_TO_DEVICE);
    crf_lut_.insert(std::make_pair(std::make_pair(op_type, data_size), d_crf));
    free(h_crf);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return d_crf;
}

} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */
