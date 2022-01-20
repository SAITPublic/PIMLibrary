/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_EXECUTOR_H_
#define _PIM_EXECUTOR_H_

#include <unordered_map>
#ifdef EMULATOR
#include "emulator/PimEmulator.h"
#endif
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
class PimExecutor
{
   public:
    PimExecutor(PimRuntimeType rt_type, PimPrecision precision);
    virtual ~PimExecutor(void) {}
    static PimExecutor* get_instance(PimRuntimeType rt_type, PimPrecision precision);

    int initialize(void);
    int deinitialize(void);
    int get_loop_counter(PimOpType op_type, int input_size);
    uint8_t* make_crf_bin(PimOpType op_type, int data_size);
    uint8_t* find_crf(PimOpType op_type, int data_size);

    int execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block);
    int execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block);
    int execute_relu(PimBo* output, PimBo* pim_data, hipStream_t stream, bool block);
    int execute_copy(PimBo* output, PimBo* pim_data, hipStream_t stream, bool block);
    int execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                   double epsilon, hipStream_t stream, bool block);
    int execute_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block);
    int execute_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, hipStream_t stream, bool block);
    int execute_custom_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, bool is_gemv_add, hipStream_t stream,
                            bool block);
    int execute_custom_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, PimBo* operand2, bool relu,
                                hipStream_t stream, bool block);
    int execute_sync(hipStream_t stream);
    int execute_dummy(void);

    int preprocess_srf(PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance, double epsilon, uint8_t* srf_binary);

   private:
    int max_crf_size_;
    pim::runtime::manager::PimManager* pim_manager_;
    std::map<std::pair<PimOpType, int>, uint8_t*> crf_lut_;
    uint8_t* d_srf_bin_buffer_;

    PimRuntimeType rt_type_;
    PimPrecision precision_;
    hipDeviceProp_t dev_prop_;
    uint8_t* pim_gemv_tmp_buffer_;
    uint8_t* zero_buffer_;
    PimBlockInfo fbi_;
    PimGemvType pim_gemv_type_;

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
    int execute_gemv_next_pim(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, hipStream_t stream,
                              bool block);
    int execute_gemv_tile_accum(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, hipStream_t stream,
                                bool block);
    int execute_gemv_tile_tree(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, hipStream_t stream,
                               bool block);
};

} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_EXECUTOR_H_ */
