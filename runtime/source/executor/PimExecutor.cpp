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
#include "executor/HIPExecutor.h"
#include "executor/OpenCLExecutor.h"
#include "executor/PimCompilerDriver.h"
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
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    static PimExecutor* instance_;
    if (rt_type == RT_TYPE_HIP) {
        instance_ = new HIPExecutor(rt_type, precision);
    } else if (rt_type == RT_TYPE_OPENCL) {
        instance_ = new OpenCLExecutor(rt_type, precision);
    } else {
        DLOG(ERROR) << "Executor for " << rt_type << " not implemented\n";
        return NULL;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return instance_;
}

int PimExecutor::initialize(void)
{
    DLOG(INFO) << "[START]" << __FUNCTION__ << "Initializing ";
    int ret = 0;
    int is_gemv_tile_tree = pim_gemv_type_ == TILE_TREE ? 1 : 0;
    pim_manager_ = pim::runtime::manager::PimManager::get_instance(rt_type_, precision_);
    pim_manager_->pim_crf_generator_->set_gemv_tile_tree(is_gemv_tile_tree);

    DLOG(INFO) << "[END]" << __FUNCTION__ << " called ";
    return ret;
}

int PimExecutor::deinitialize(void)
{
    int ret = 0;
    DLOG(INFO) << "[END]" << __FUNCTION__ << " called";
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

void* PimExecutor::make_crf_bin(PimOpType op_type, int data_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    uint8_t* h_crf = new uint8_t[max_crf_size_];
    uint8_t* d_crf;
    int crf_size;
    pim_manager_->alloc_memory((void**)&d_crf, max_crf_size_, MEM_TYPE_DEVICE);

    int lc = get_loop_counter(op_type, data_size);
    pim_manager_->pim_crf_generator_->gen_binary_with_loop(op_type, lc, h_crf, &crf_size);

    pim_manager_->copy_memory((void*)d_crf, (void*)h_crf, max_crf_size_, HOST_TO_DEVICE);
    crf_lut_.insert(std::make_pair(std::make_pair(op_type, data_size), d_crf));
    free(h_crf);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return (void*)d_crf;
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
} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */
