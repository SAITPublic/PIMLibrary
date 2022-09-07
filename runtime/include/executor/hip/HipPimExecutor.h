/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _HIP_PIM_EXECUTOR_H_
#define _HIP_PIM_EXECUTOR_H_

#include "emulator/PimEmulator.h"
#include "executor/IPimExecutor.h"
#include "executor/PimCrfBinGen.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace executor
{
class HipPimExecutor : public IPimExecutor
{
   public:
    HipPimExecutor(pim::runtime::manager::PimManager* pim_manager, PimPrecision precision);
    virtual ~HipPimExecutor(void);

    int initialize(void);
    int deinitialize(void);
    int execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block);
    int execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block);
    int execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block);
    int execute_copy(PimBo* output, PimBo* pim_data, void* stream, bool block);
    int execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                   double epsilon, void* stream, bool block);
    int execute_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                     bool block);
    int execute_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block);
    int execute_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block);
    int execute_gemv_list(PimBo* output, PimBo* input, PimBo* weight, void* stream, bool block);
    int execute_gemv_list_normal(PimBo* output, PimBo* input, PimBo* weight, void* stream, bool block);
    int execute_gemv_list_chwise(PimBo* output, PimBo* input, PimBo* weight, int ch_per_op, void* stream, bool block);
    int execute_custom_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, bool is_gemv_add, void* stream,
                            bool block);
    int execute_custom_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, PimBo* operand2, bool relu,
                                void* stream, bool block);
    int execute_sync(void* stream);
    int execute_dummy(void);
    void* createStream(void);

   private:
    int execute_gemv_next_pim(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, void* stream,
                              bool block);
    int execute_gemv_tile_accum(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, void* stream,
                                bool block);
    int execute_gemv_tile_tree(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, void* stream,
                               bool block);
    int execute_aligned_gemm_tile_accum(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                                        void* stream, bool block);
    int execute_chwise_gemm_tile_accum(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                                       void* stream, bool block);

   private:
    pim::runtime::manager::PimManager* pim_manager_;
    std::shared_ptr<pim::runtime::manager::PimDevice> pim_device_;
    PimCrfBinGen* pim_crf_generator_;
    PimPrecision precision_;
    PimBlockInfo* pbi_;
    PimGemvType pim_gemv_type_;
    int max_crf_size_;
    uint8_t* d_srf_bin_buffer_;
    uint8_t* pim_gemv_tmp_buffer_;
    uint8_t* zero_buffer_;
    hipDeviceProp_t dev_prop_;

#ifdef EMULATOR
    PimMemTraceData* d_fmtd16_;
    int* d_fmtd16_size_;
    pim::runtime::emulator::PimEmulator* pim_emulator_;
    PimMemTraceData* h_fmtd16_;
    PimMemTraceData* h_fmtd32_;
    PimMemTracer* d_emulator_trace_;
    int* h_fmtd16_size_;
    int* h_fmtd32_size_;
    int fmtd_size_per_ch_;
    int max_block_size_;
    int max_fmtd_size_;
#endif
};
}  // namespace executor
}  // namespace runtime
}  // namespace pim

#endif
