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

#include <map>
#ifdef EMULATOR
#include "emulator/PimEmulator.h"
#endif
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
#include "pim_data_types.h"

#define MAX_NUM_GPUS 10

namespace pim
{
namespace runtime
{
namespace executor
{
constexpr uint32_t compiler_env_value = PIM_COMPILER_ENABLE;

class PimExecutor
{
   public:
    PimExecutor(PimRuntimeType rt_type, PimPrecision precision);
    virtual ~PimExecutor(void) {}
    static PimExecutor* get_instance(PimRuntimeType rt_type, PimPrecision precision);

    virtual int initialize(void);
    virtual int deinitialize(void);
    int get_loop_counter(PimOpType op_type, int input_size);
    void* make_crf_bin(PimOpType op_type, int data_size);
    uint8_t* find_crf(PimOpType op_type, int data_size);
    virtual void* createStream() = 0;

    virtual int execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block) = 0;
    virtual int execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block) = 0;
    virtual int execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block) = 0;
    virtual int execute_copy(PimBo* output, PimBo* pim_data, void* stream, bool block) = 0;
    virtual int execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                           double epsilon, void* stream, bool block) = 0;
    virtual int execute_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                             bool block) = 0;
    virtual int execute_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block) = 0;
    virtual int execute_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block) = 0;
    virtual int execute_gemv_list(PimBo* output, PimBo* input, PimBo* weight, void* stream, bool block) = 0;
    virtual int execute_gemv_list_normal(PimBo* output, PimBo* input, PimBo* weight, void* stream, bool block) = 0;
    virtual int execute_gemv_list_chwise(PimBo* output, PimBo* input, PimBo* weight, int ch_per_op, void* stream,
                                         bool block) = 0;
    virtual int execute_custom_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, bool is_gemv_add, void* stream,
                                    bool block) = 0;
    virtual int execute_custom_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, PimBo* operand2, bool relu,
                                        void* stream, bool block) = 0;
    virtual int execute_sync(void* stream) = 0;
    virtual int execute_dummy(void) = 0;

    int preprocess_srf(PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance, double epsilon, uint8_t* srf_binary);

   protected:
    int max_crf_size_;
    pim::runtime::manager::PimManager* pim_manager_;
    std::shared_ptr<pim::runtime::manager::PimDevice> pim_device_;
    std::map<std::pair<PimOpType, int>, uint8_t*> crf_lut_;

    PimRuntimeType rt_type_;
    PimPrecision precision_;
    PimBlockInfo* pbi_;
    PimGemvType pim_gemv_type_;
    PimExecutor* pim_executor_;

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

} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_EXECUTOR_H_ */
