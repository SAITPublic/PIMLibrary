/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_EXECUTOR_H_
#define _FIM_EXECUTOR_H_

#include <unordered_map>
#include "emulator/FimEmulator.h"
#include "fim_data_types.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "manager/FimInfo.h"
#include "manager/FimManager.h"

namespace fim
{
namespace runtime
{
namespace executor
{
class FimExecutor
{
   public:
    FimExecutor(FimRuntimeType rt_type, FimPrecision precision);
    virtual ~FimExecutor(void) {}
    static FimExecutor* get_instance(FimRuntimeType rt_type, FimPrecision precision);

    int initialize(void);
    int deinitialize(void);
    int get_loop_counter(FimOpType op_type, int input_size);
    uint8_t* make_crf_bin(FimOpType op_type, int data_size);
    uint8_t* find_crf(FimOpType op_type, int data_size);

    int execute_add(FimBo* output, FimBo* operand0, FimBo* operand1, hipStream_t stream, bool block);
    int execute_mul(FimBo* output, FimBo* operand0, FimBo* operand1, hipStream_t stream, bool block);
    int execute_relu(FimBo* output, FimBo* fim_data, hipStream_t stream, bool block);
    int execute_gemv(FimBo* output, FimBo* operand0, FimBo* operand1, hipStream_t stream, bool block);
    int execute_gemv_add(FimBo* output, FimBo* operand0, FimBo* operand1, hipStream_t stream, bool block);
    int execute_bn(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance,
                   double epsilon, hipStream_t stream, bool block);
    int execute_sync(hipStream_t stream);
    int execute_dummy(void);

    int preprocess_srf(FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance, double epsilon, uint8_t* srf_binary);

   private:
    int max_crf_size_;
    fim::runtime::manager::FimManager* fim_manager_;
    std::map<std::pair<FimOpType, int>, uint8_t*> crf_lut_;
    uint8_t* d_srf_bin_buffer_;

    FimRuntimeType rt_type_;
    FimPrecision precision_;
    hipDeviceProp_t dev_prop_;
    uint8_t* fim_gemv_tmp_buffer_;
    FimBlockInfo fbi_;
#ifdef EMULATOR
    FimMemTraceData* d_fmtd16_;
    int* d_fmtd16_size_;
    fim::runtime::emulator::FimEmulator* fim_emulator_;
    FimMemTraceData* h_fmtd16_;
    FimMemTraceData* h_fmtd32_;
    int* h_fmtd16_size_;
    int* h_fmtd32_size_;
    int fmtd_size_per_ch_;
    int max_block_size_;
    int max_fmtd_size_;
#endif
};

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_EXECUTOR_H_ */
