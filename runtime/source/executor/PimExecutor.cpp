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
#include "executor/PimCompilerDriver.h"
#include "executor/gpu_hip_kernels/gpu_custom_ops.h"
#include "executor/pim_hip_kernels/pim_op_kernels.pimk"
#include "hip/hip_runtime.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

extern uint64_t g_pim_base_addr[MAX_NUM_GPUS];
namespace pim
{
namespace runtime
{
namespace executor
{
constexpr uint32_t compiler_env_value = PIM_COMPILER_ENABLE;

PimExecutor::PimExecutor(PimRuntimeType rt_type, PimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";
    get_pim_block_info(&fbi_);
#ifdef EMULATOR
    pim_emulator_ = pim::runtime::emulator::PimEmulator::get_instance();
    fmtd_size_per_ch_ = 100000;
    max_block_size_ = fbi_.num_pim_chan;
    max_fmtd_size_ = fmtd_size_per_ch_ * max_block_size_;
#endif
    pim_gemv_type_ = TILE_ACCUM;

    const char* env_p = std::getenv("ENABLE_NEXT_PIM");
    if (env_p != nullptr) {
        if (env_p[0] == '1') {
            pim_gemv_type_ = NEXT_PIM;
            std::cout << "NEXT_PIM(GEMV) is enabled" << std::endl;
        }
    }
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
    hipGetDevice(&device_id);
    hipGetDeviceProperties(&dev_prop_, device_id);
    DLOG(INFO) << " System minor " << dev_prop_.minor << std::endl;
    DLOG(INFO) << " System major " << dev_prop_.major << std::endl;
    DLOG(INFO) << " agent prop name " << dev_prop_.name << std::endl;
    DLOG(INFO) << " device id " << device_id << std::endl;
    DLOG(INFO) << " hip Device prop succeeded " << std::endl;

    int is_gemv_tile_tree = pim_gemv_type_ == TILE_TREE ? 1 : 0;
    pim_manager_ = pim::runtime::manager::PimManager::get_instance(rt_type_, precision_);
    pim_manager_->pim_crf_generator_->set_gemv_tile_tree(is_gemv_tile_tree);

    max_crf_size_ = 128;
    int max_srf_size = 2048;

    hipMalloc((void**)&d_srf_bin_buffer_, max_srf_size);
    hipMalloc((void**)&zero_buffer_, 32);
    hipMemset(zero_buffer_, 0, 32);

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
    pim_manager_->alloc_memory((void**)&pim_gemv_tmp_buffer_, 8 * 2 * 1024 * 1024, MEM_TYPE_PIM);

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
    hipFree((void*)zero_buffer_);
#ifdef EMULATOR
    hipFree((void*)d_fmtd16_);
    hipFree((void*)d_fmtd16_size_);
    free(h_fmtd16_);
    free(h_fmtd16_size_);
    free(h_fmtd32_);
    free(h_fmtd32_size_);
#endif
    pim_manager_->free_memory((void*)pim_gemv_tmp_buffer_, MEM_TYPE_PIM);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::get_loop_counter(PimOpType op_type, int input_size)
{
    int lc = 0;

    int num_transaction = (input_size / 16) / sizeof(uint16_t);
    int num_parallelism = fbi_.num_pim_blocks * fbi_.num_pim_chan * fbi_.num_pim_rank * fbi_.num_grf;
    int num_tile = num_transaction / num_parallelism;

    if (op_type == OP_GEMV) {
        if (pim_gemv_type_ == TILE_TREE)
            lc = (input_size / fbi_.trans_size / fbi_.num_grf_A / 2) - 1;
        else
            lc = input_size / fbi_.trans_size / fbi_.num_grf_A;
    } else
        lc = num_tile / 2 - 1;

    return lc;
}

int PimExecutor::execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (compiler_env_value == 1) {
        pimc_driver::PimCDriver elt_op_execute;

        std::vector<uint32_t> dims{output->bshape.n, output->bshape.c, output->bshape.h, output->bshape.w};
        pimc_driver::Tensor<half_float::half> input0_t(elt_op_execute.create_tensor_desc(dims),
                                                       (half_float::half*)operand0->data);
        pimc_driver::Tensor<half_float::half> input1_t(elt_op_execute.create_tensor_desc(dims),
                                                       (half_float::half*)operand1->data);
        pimc_driver::Tensor<half_float::half> output_t(elt_op_execute.create_tensor_desc(dims),
                                                       (half_float::half*)output->data);

        std::vector<pimc::TensorDesc> inputs{input0_t.get_desc(), input1_t.get_desc()};
        std::vector<pimc::TensorDesc> outputs{output_t.get_desc()};

        auto pim_op = elt_op_execute.generate_code(OP_ELT_ADD, inputs, outputs);
        auto kernel = elt_op_execute.compile_code();

        pimc_driver::EltArgs<half_float::half> kargs(&input0_t, &input1_t, &output_t, pim_op->get_crf_binary(), kernel);

        elt_op_execute.execute_code(&kargs);
        if (block) hipStreamSynchronize(nullptr);
    } else {
        int output_size = output->size;

        uint8_t* crf_bin = find_crf(OP_ELT_ADD, output_size);
        int crf_size = 32;
        if (crf_bin == nullptr) {
            crf_bin = make_crf_bin(OP_ELT_ADD, output_size);
        }

        int align_size = (131072 << 1);
        int num_tile = (output_size + align_size - 1) / align_size;

        unsigned blocks = 64;
        unsigned threads_per_block = 32;
        int device_id;
        hipGetDevice(&device_id);
        hipLaunchKernelGGL(
            elt_op_pim, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)operand0->data,
            (uint8_t*)operand1->data, (uint8_t*)(g_pim_base_addr[device_id]), (uint8_t*)output->data, num_tile,
#ifdef EMULATOR
            (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_, (PimMemTracer*)d_emulator_trace_,
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
        pim_emulator_->execute_elt_op(output, operand0, operand1, h_fmtd32_, h_fmtd32_size_[0],
                                      g_pim_base_addr[device_id]);
#else
        if (block) hipStreamSynchronize(stream);
#endif
    }
    return ret;
}

int PimExecutor::execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (compiler_env_value == 1) {
        pimc_driver::PimCDriver elt_op_execute;

        std::vector<uint32_t> dims{output->bshape.n, output->bshape.c, output->bshape.h, output->bshape.w};
        pimc_driver::Tensor<half_float::half> input0_t(elt_op_execute.create_tensor_desc(dims),
                                                       (half_float::half*)operand0->data);
        pimc_driver::Tensor<half_float::half> input1_t(elt_op_execute.create_tensor_desc(dims),
                                                       (half_float::half*)operand1->data);
        pimc_driver::Tensor<half_float::half> output_t(elt_op_execute.create_tensor_desc(dims),
                                                       (half_float::half*)output->data);

        std::vector<pimc::TensorDesc> inputs{input0_t.get_desc(), input1_t.get_desc()};
        std::vector<pimc::TensorDesc> outputs{output_t.get_desc()};

        auto pim_op = elt_op_execute.generate_code(OP_ELT_MUL, inputs, outputs);
        auto kernel = elt_op_execute.compile_code();

        pimc_driver::EltArgs<half_float::half> kargs(&input0_t, &input1_t, &output_t, pim_op->get_crf_binary(), kernel);

        elt_op_execute.execute_code(&kargs);
        if (block) hipStreamSynchronize(nullptr);
    } else {
        int output_size = output->size;

        uint8_t* crf_bin = find_crf(OP_ELT_MUL, output->size);
        int crf_size = 32;
        if (crf_bin == nullptr) {
            crf_bin = make_crf_bin(OP_ELT_MUL, output->size);
        }

        int align_size = (131072 << 1);
        int num_tile = (output_size + align_size - 1) / align_size;

        unsigned blocks = 64;
        unsigned threads_per_block = 32;
        int device_id;
        hipGetDevice(&device_id);
        hipLaunchKernelGGL(
            elt_op_pim, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)operand0->data,
            (uint8_t*)operand1->data, (uint8_t*)(g_pim_base_addr[device_id]), (uint8_t*)output->data, num_tile,
#ifdef EMULATOR
            (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_, (PimMemTracer*)d_emulator_trace_,
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
        pim_emulator_->execute_elt_op(output, operand0, operand1, h_fmtd32_, h_fmtd32_size_[0],
                                      g_pim_base_addr[device_id]);
#else
        if (block) hipStreamSynchronize(stream);
#endif
    }
    return ret;
}

int PimExecutor::execute_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    int ret = 0;
    int is_gemv_add = 0;
    switch (pim_gemv_type_) {
        case NEXT_PIM:
            ret = execute_gemv_next_pim(output, operand0, operand1, is_gemv_add, stream, block);
            break;
        case TILE_TREE:
            ret = execute_gemv_tile_tree(output, operand0, operand1, is_gemv_add, stream, block);
            break;
        case TILE_ACCUM:
        default:
            ret = execute_gemv_tile_accum(output, operand0, operand1, is_gemv_add, stream, block);
            break;
    }
    return ret;
}

int PimExecutor::execute_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block)
{
    int ret = 0;
    int is_gemv_add = 1;
    switch (pim_gemv_type_) {
        case NEXT_PIM:
            ret = execute_gemv_next_pim(output, operand0, operand1, is_gemv_add, stream, block);
            break;
        case TILE_TREE:
        /* TODO : add tile tree function */
        case TILE_ACCUM:
        default:
            ret = execute_gemv_tile_accum(output, operand0, operand1, is_gemv_add, stream, block);
            break;
    }
    return ret;
}

int PimExecutor::execute_gemv_tile_accum(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add,
                                         hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

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

    if (compiler_env_value == 1) {
        pimc_driver::PimCDriver gemv_execute;

        std::vector<uint32_t> in_dims{operand0->bshape.n, operand0->bshape.c, operand0->bshape.h, operand0->bshape.w};
        std::vector<uint32_t> wt_dims{operand1->bshape.n, operand1->bshape.c, operand1->bshape_r.h,
                                      operand1->bshape_r.w};
        std::vector<uint32_t> out_dims{output->bshape.n, output->bshape.c, output->bshape.h, output->bshape.w};

        pimc_driver::Tensor<half_float::half> inputs_t(gemv_execute.create_tensor_desc(in_dims),
                                                       (half_float::half*)operand0->data);
        pimc_driver::Tensor<half_float::half> weights_t(gemv_execute.create_tensor_desc(wt_dims),
                                                        (half_float::half*)operand1->data);
        pimc_driver::Tensor<half_float::half> output_t(gemv_execute.create_tensor_desc(out_dims),
                                                       (half_float::half*)output->data);

        std::vector<pimc::TensorDesc> inputs{inputs_t.get_desc(), weights_t.get_desc()};
        std::vector<pimc::TensorDesc> outputs{output_t.get_desc()};

        auto pim_op = gemv_execute.generate_code(OP_GEMV, inputs, outputs);
        auto kernel = gemv_execute.compile_code();

        pimc_driver::GemvKArgs<half_float::half> kargs(&inputs_t, &output_t, &weights_t, pim_op->get_crf_binary(),
                                                       kernel);
        kargs.set_compute_tile(n_compute_tile);
        kargs.set_memory_tile(n_memory_tile);
        kargs.set_out_tile(n_out_tile);
        kargs.set_gemv_add(is_gemv_add);
        kargs.set_temp_buffer(pim_gemv_tmp_buffer_);

        gemv_execute.execute_code(&kargs);
        if (block) hipStreamSynchronize(nullptr);
    } else {
        PIM_PROFILE_TICK(CreateCRFBin);

        void (*gemv_kernel)(volatile uint8_t * __restrict__, volatile uint8_t * __restrict__,
                            volatile uint8_t * __restrict__, volatile uint8_t * __restrict__,
                            volatile uint8_t * __restrict__, int, int, int, int, int,
#ifdef EMULATOR
                            PimMemTraceData*, int*, int, PimMemTracer*,
#endif
                            uint8_t*, int, int);

        switch (n_compute_tile) {
            case 8:
                gemv_kernel = gemv_pim_64cu_64th_8tile_fp16;
                break;
            default:
                gemv_kernel = gemv_pim_64cu_64th_fp16;
                break;
        }

        /* TODO: check tile_accum crf bin */
        uint8_t* crf_bin = find_crf(OP_GEMV, compute_size * sizeof(uint16_t));
        int crf_size = 32;
        if (crf_bin == nullptr) {
            crf_bin = make_crf_bin(OP_GEMV, compute_size * sizeof(uint16_t));
        }
        PIM_PROFILE_TOCK(CreateCRFBin);

        PIM_PROFILE_TICK(RunGemvKernel);
        int device_id;
        hipGetDevice(&device_id);
        hipLaunchKernelGGL(
            gemv_kernel, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)(g_pim_base_addr[device_id]),
            (uint8_t*)weight->data, (uint8_t*)pim_gemv_tmp_buffer_, (uint8_t*)input->data, (uint8_t*)output->data,
            n_batch, n_memory_tile, n_compute_tile, n_out_tile, real_out_size,
#ifdef EMULATOR
            (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_, (PimMemTracer*)d_emulator_trace_,
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

        pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                         OP_GEMV);
        if (is_gemv_add)
            pim_emulator_->execute_gemv_add_tile_accum(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV,
                                                       g_pim_base_addr[device_id], pim_gemv_tmp_buffer_);
        else
            pim_emulator_->execute_gemv_tile_accum(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV,
                                                   g_pim_base_addr[device_id], pim_gemv_tmp_buffer_);

        PIM_PROFILE_TOCK(RunGemvEmulation);
#endif
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::execute_gemv_tile_tree(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add,
                                        hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

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
    int device_id;
    hipGetDevice(&device_id);
    hipLaunchKernelGGL(gemv_tree_pim_64cu_64th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)g_pim_base_addr[device_id], (uint8_t*)weight->data, (uint8_t*)pim_gemv_tmp_buffer_,
                       (uint8_t*)zero_buffer_, (uint8_t*)input->data, (uint8_t*)output->data, n_batch, n_memory_tile,
                       n_memory_tile, n_out_tile, real_out_size,
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
    pim_emulator_->execute_gemv_tile_tree(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV,
                                          g_pim_base_addr[device_id], pim_gemv_tmp_buffer_);
    PIM_PROFILE_TOCK(RunGemvEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::execute_gemv_list(PimBo* output, PimBo* input, PimBo* weight, hipStream_t stream, bool block)
{
    int ret = 0;
    bool is_chwise = false;
    int ch_per_batch = 0;

    if (weight->bshape.n % fbi_.num_pim_chan == 0) {
        is_chwise = true;
        ch_per_batch = 1;
    }

    if (is_chwise == true) {
        ret = execute_gemv_list_chwise(output, input, weight, ch_per_batch, stream, block);
    } else {
        ret = execute_gemv_list_normal(output, input, weight, stream, block);
    }

    return ret;
}

int PimExecutor::execute_gemv_list_normal(PimBo* output, PimBo* input, PimBo* weight, hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    unsigned blocks = fbi_.num_pim_chan;
    unsigned threads_per_block = 64;
    int list_size = output->bshape.n;

    int input_size = 128 * ceil((float)weight->bshape_r.w / 128);
    if (input_size < 256) input_size = 256;
    int output_size = weight->bshape.h * weight->bshape.n;
    int n_in_tile = input_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_out_tile = output_size / (fbi_.num_pim_chan * fbi_.num_pim_blocks * fbi_.num_grf_B);
    int is_gemv_add = 0;

    PIM_PROFILE_TICK(CreateCRFBin);

    void (*gemv_kernel)(volatile uint8_t * __restrict__, volatile uint8_t * __restrict__,
                        volatile uint8_t * __restrict__, volatile uint8_t * __restrict__,
                        volatile uint8_t * __restrict__, int, int, int, int, int, int,
#ifdef EMULATOR
                        PimMemTraceData*, int*, int, PimMemTracer*,
#endif
                        uint8_t*, int, int);

    gemv_kernel = gemv_list_pim_64cu_64th_fp16;

    uint8_t* crf_bin = find_crf(OP_GEMV, input_size * sizeof(uint16_t));
    int crf_size = 32;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_GEMV, input_size * sizeof(uint16_t));
    }
    PIM_PROFILE_TOCK(CreateCRFBin);
    PIM_PROFILE_TICK(RunGemvListKernel);

    int device_id;
    hipGetDevice(&device_id);
    hipLaunchKernelGGL(
        gemv_kernel, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)g_pim_base_addr[device_id],
        (uint8_t*)weight->data, (uint8_t*)pim_gemv_tmp_buffer_, (uint8_t*)input->data, (uint8_t*)output->data, 1,
        input_size, output_size, n_in_tile, n_out_tile, list_size,
#ifdef EMULATOR
        (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_, (PimMemTracer*)d_emulator_trace_,
#endif
        (uint8_t*)crf_bin, crf_size, is_gemv_add);

#ifndef EMULATOR
    if (block) hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvListKernel);
#endif

#ifdef EMULATOR
    hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvListKernel);

    PIM_PROFILE_TICK(RunGemvListEmulation);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;

    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    if (is_gemv_add)
        pim_emulator_->execute_gemv_add_tile_accum(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV,
                                                   g_pim_base_addr[device_id], pim_gemv_tmp_buffer_);
    else
        pim_emulator_->execute_gemv_tile_accum(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV,
                                               g_pim_base_addr[device_id], pim_gemv_tmp_buffer_);

    PIM_PROFILE_TOCK(RunGemvListEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";

    return ret;
}

int PimExecutor::execute_gemv_list_chwise(PimBo* output, PimBo* input, PimBo* weight, int ch_per_op, hipStream_t stream,
                                          bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    unsigned blocks = fbi_.num_pim_chan;
    unsigned threads_per_block = 64;
    int list_size = output->bshape.n;

    int input_size = 128 * ceil((float)weight->bshape_r.w / 128);
    if (input_size < 256) input_size = 256;
    int output_size = weight->bshape.h;
    int n_in_tile = input_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_out_tile = output_size / (ch_per_op * fbi_.num_pim_blocks * fbi_.num_grf_B);
    int is_gemv_add = 0;

    PIM_PROFILE_TICK(CreateCRFBin);

    void (*gemv_kernel)(volatile uint8_t * __restrict__, volatile uint8_t * __restrict__,
                        volatile uint8_t * __restrict__, volatile uint8_t * __restrict__,
                        volatile uint8_t * __restrict__, int, int, int, int, int, int,
#ifdef EMULATOR
                        PimMemTraceData*, int*, int, PimMemTracer*,
#endif
                        uint8_t*, int, int, int);
    gemv_kernel = gemv_list_chwise_pim_64cu_64th_fp16;

    uint8_t* crf_bin = find_crf(OP_GEMV, input_size * sizeof(uint16_t));
    int crf_size = 32;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_GEMV, input_size * sizeof(uint16_t));
    }
    PIM_PROFILE_TOCK(CreateCRFBin);
    PIM_PROFILE_TICK(RunGemvListKernel);

    int device_id;
    hipGetDevice(&device_id);
    hipLaunchKernelGGL(
        gemv_kernel, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)g_pim_base_addr[device_id],
        (uint8_t*)weight->data, (uint8_t*)pim_gemv_tmp_buffer_, (uint8_t*)input->data, (uint8_t*)output->data, 1,
        input_size, output_size, n_in_tile, n_out_tile, list_size,
#ifdef EMULATOR
        (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_, (PimMemTracer*)d_emulator_trace_,
#endif
        (uint8_t*)crf_bin, crf_size, ch_per_op, is_gemv_add);

#ifndef EMULATOR
    if (block) hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvListKernel);
#endif

#ifdef EMULATOR
    hipStreamSynchronize(stream);
    PIM_PROFILE_TOCK(RunGemvListKernel);

    PIM_PROFILE_TICK(RunGemvListEmulation);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;

    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    if (is_gemv_add)
        pim_emulator_->execute_gemv_add_tile_accum(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV,
                                                   g_pim_base_addr[device_id], pim_gemv_tmp_buffer_);
    else
        pim_emulator_->execute_gemv_tile_accum(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV,
                                               g_pim_base_addr[device_id], pim_gemv_tmp_buffer_);

    PIM_PROFILE_TOCK(RunGemvListEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";

    return ret;
}

int PimExecutor::execute_relu(PimBo* output, PimBo* pim_data, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (compiler_env_value == 1) {
        pimc_driver::PimCDriver relu_execute;

        std::vector<uint32_t> dims{output->bshape.n, output->bshape.c, output->bshape.h, output->bshape.w};
        pimc_driver::Tensor<half_float::half> input_t(relu_execute.create_tensor_desc(dims),
                                                      (half_float::half*)pim_data->data);
        pimc_driver::Tensor<half_float::half> output_t(relu_execute.create_tensor_desc(dims),
                                                       (half_float::half*)output->data);

        std::vector<pimc::TensorDesc> inputs{input_t.get_desc()};
        std::vector<pimc::TensorDesc> outputs{output_t.get_desc()};

        auto pim_op = relu_execute.generate_code(OP_RELU, inputs, outputs);
        auto kernel = relu_execute.compile_code();

        pimc_driver::ReluArgs<half_float::half> kargs(&input_t, &output_t, pim_op->get_crf_binary(), kernel);
        auto out_dim = (output_t.get_desc().get_dim(3) * sizeof(half_float::half)) / 32;
        uint32_t num_tile = out_dim / ((fbi_.num_pim_blocks * fbi_.num_pim_chan * fbi_.num_grf) / 2);
        kargs.set_num_tile(num_tile);

        relu_execute.execute_code(&kargs);
    } else {
        uint8_t* crf_bin = find_crf(OP_RELU, output->size);
        int crf_size = 32;
        if (crf_bin == nullptr) {
            crf_bin = make_crf_bin(OP_RELU, output->size);
        }

        unsigned blocks = fbi_.num_pim_chan;
        unsigned threads_per_block = 32;
        int device_id;
        hipGetDevice(&device_id);
        hipLaunchKernelGGL(relu_pim, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)pim_data->data,
                           (uint8_t*)(g_pim_base_addr[device_id]), (uint8_t*)output->data, (int)output->size,
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
                                                         OP_RELU);
        pim_emulator_->execute_relu(output, pim_data, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr[device_id]);
#else
        if (block) hipStreamSynchronize(stream);
#endif
    }
    return ret;
}

int PimExecutor::execute_copy(PimBo* output, PimBo* pim_data, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (compiler_env_value == 1) {
        DLOG(ERROR) << "PIM compiler path is not supported yet";
    } else {
        uint8_t* crf_bin = find_crf(OP_COPY, output->size);
        int crf_size = 32;
        if (crf_bin == nullptr) {
            crf_bin = make_crf_bin(OP_COPY, output->size);
        }

        unsigned blocks = fbi_.num_pim_chan;
        unsigned threads_per_block = 32;

        int device_id;
        hipGetDevice(&device_id);
        hipLaunchKernelGGL(copy_pim, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)pim_data->data,
                           (uint8_t*)g_pim_base_addr[device_id], (uint8_t*)output->data, (int)output->size,
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
                                                         OP_RELU);
        pim_emulator_->execute_copy(output, pim_data, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr[device_id]);
#else
        if (block) hipStreamSynchronize(stream);
#endif
    }
    return ret;
}

int PimExecutor::execute_gemv_next_pim(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add,
                                       hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

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

    /* TODO: check tile_accum crf bin */
    uint8_t* crf_bin = find_crf(OP_GEMV, compute_size * sizeof(uint16_t));
    int crf_size = 32;
    if (crf_bin == nullptr) {
        crf_bin = make_crf_bin(OP_GEMV, compute_size * sizeof(uint16_t));
    }
    PIM_PROFILE_TOCK(CreateCRFBin);

    PIM_PROFILE_TICK(RunGemvKernel);
    int device_id;
    hipGetDevice(&device_id);
    hipLaunchKernelGGL(gemv_next_pim_64cu_64th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)g_pim_base_addr[device_id], (uint8_t*)weight->data, (uint8_t*)pim_gemv_tmp_buffer_,
                       (uint8_t*)input->data, (uint8_t*)output->data, n_batch, n_memory_tile, n_compute_tile,
                       n_out_tile, real_out_size,
#ifdef EMULATOR
                       (PimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                       (PimMemTracer*)d_emulator_trace_,
#endif
                       (uint8_t*)crf_bin, crf_size, is_gemv_add);
#ifndef EMULATOR
    if (block) hipStreamSynchronize(stream);
#endif
#ifdef EMULATOR
/* TODO:verify Emulator Path */
#endif
    PIM_PROFILE_TOCK(RunGemvKernel);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
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
    int device_id;
    hipGetDevice(&device_id);
    hipLaunchKernelGGL(bn_pim_nr_sip, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)pim_data->data,
                       (uint8_t*)g_pim_base_addr[device_id], (uint8_t*)pim_gemv_tmp_buffer_, (uint8_t*)output->data,
                       (int)num_tile, output->bshape.n, output->bshape.c, output->bshape.w,
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
    pim_emulator_->execute_bn(output, pim_data, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr[device_id],
                              pim_gemv_tmp_buffer_);
#else
    if (block) hipStreamSynchronize(stream);
#endif
    delete[] srf_binary;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::execute_custom_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, bool is_gemv_add,
                                     hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    uint32_t m, n, k;

    if (operand0->bshape_r.n != 1) {
        std::cout << "[Error] " << __FUNCTION__ << ": GEMM is not supported" << std::endl;
        return 1;
    }

    // if operand1 is on HOST, copy it to DEVICE
    bool copy_to_device = false;
    if (operand1->mem_type == MEM_TYPE_HOST) {
        copy_to_device = true;
        PimBShape bs = operand1->bshape;
        PimBo* operand1_device = PimCreateBo(bs.w, bs.h, bs.c, bs.n, PIM_FP16, MEM_TYPE_DEVICE, 0);
        pim_manager_->copy_memory(operand1_device->data, operand1->data, operand1->size, HOST_TO_DEVICE);
        operand1 = operand1_device;
    }

    void* vec = operand0->data;
    void* mat = operand1->data;
    void* out = output->data;

    float alpha = 1.0f;
    float beta = is_gemv_add ? 1.0f : 0.0f;

    if (is_transposed(operand1)) {
        m = operand1->bshape_r.h;
        k = operand1->bshape_r.w;
        n = 1;
        rocblas_gemv_fp16_Axy(mat, vec, out, m, n, k, alpha, beta, stream);
    } else {
        m = 1;
        k = operand1->bshape_r.w;
        n = operand1->bshape_r.h;
        rocblas_gemv_fp16_xAy(vec, mat, out, m, n, k, alpha, beta, stream);
    }

    if (copy_to_device) {
        PimDestroyBo(operand1);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecutor::execute_custom_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, PimBo* operand2, bool relu,
                                         hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    uint32_t m, n, k;

    if (operand0->bshape_r.n != 1) {
        std::cout << "[Error] " << __FUNCTION__ << ": GEMM is not supported" << std::endl;
        return 1;
    }

    void* vec = operand0->data;
    void* mat = operand1->data;
    void* in = operand2->data;
    void* out = output->data;

    float alpha = 1.0f;
    float beta = 0.0f;

    if (is_transposed(operand1)) {
        m = operand1->bshape_r.h;
        k = operand1->bshape_r.w;
        n = 1;
        if (m == 32317) {
            rocblas_addmv_fp16_Axy_large(in, mat, vec, out, m, n, k, alpha, beta, relu, stream);
        } else {
            rocblas_addmv_fp16_Axy(in, mat, vec, out, m, n, k, alpha, beta, relu, stream);
        }
    } else {
        m = 1;
        k = operand1->bshape_r.w;
        n = operand1->bshape_r.h;
        rocblas_addmv_fp16_xAy(in, vec, mat, out, m, n, k, alpha, beta, relu, stream);
    }

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
