/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _HIP_EXECUTOR_H_
#define _HIP_EXECUTOR_H_

#include "PimExecutor.h"
#include "emulator/PimEmulator.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace executor
{
class HIPExecutor : public PimExecutor
{
   public:
    HIPExecutor(PimRuntimeType rt_type, PimPrecision precision);
    virtual ~HIPExecutor(void) {}
    int initialize(void);
    int deinitialize(void);

    void* createStream();
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

   private:
    uint8_t* d_srf_bin_buffer_;
    uint8_t* pim_gemv_tmp_buffer_;
    uint8_t* zero_buffer_;
    hipDeviceProp_t dev_prop_;

    int execute_gemv_next_pim(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, void* stream,
                              bool block);
    int execute_gemv_tile_accum(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, void* stream,
                                bool block);
    int execute_gemv_tile_tree(PimBo* output, PimBo* operand0, PimBo* operand1, int is_gemv_add, void* stream,
                               bool block);
    int execute_gemm_tile_accum(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                                void* stream, bool block);
};
}  // namespace executor
}  // namespace runtime
}  // namespace pim

#endif
