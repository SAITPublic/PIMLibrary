/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#ifndef _OCL_PIM_EXECUTOR_H_
#define _OCL_PIM_EXECUTOR_H_

#include <CL/cl.h>
#include "PimRuntime.h"
#include "emulator/ocl/OclPimEmulator.h"
#include "executor/IPimExecutor.h"
#include "executor/PimCrfBinGen.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
#include "manager/ocl/OclMemoryManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace executor
{
class OclPimExecutor : public IPimExecutor
{
   public:
    OclPimExecutor(pim::runtime::manager::PimManager* pim_manager, pim::runtime::PimRuntime* pim_runtime_,
                   PimPrecision precision);
    virtual ~OclPimExecutor(void);

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
    int execute_custom_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, bool is_gemv_add, void* stream,
                            bool block);
    int execute_custom_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, PimBo* operand2, bool relu,
                                void* stream, bool block);

    int execute_sync(void* stream) { return -1; }
    int execute_dummy(void) { return -1; }
    void* createStream(void) { return nullptr; }
    void set_gemm_order(PimGemmOrder gemm_order) { gemm_order_ = gemm_order; }

   private:
    int check_cl_program_path(void);
    int build_cl_program_with_source(void);
    int save_cl_program_binary(void);
    int build_cl_program_with_binary(void);
    std::string load_cl_file(std::string filename);

    int execute_eltwise(PimOpType eltop, PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block);
    int execute_aligned_gemm_tile_accum(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                                        void* stream, bool block);
    int execute_chwise_gemm_tile_accum(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                                       void* stream, bool block);
    int execute_ocl_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                         bool block);
    int execute_gemv(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                     bool block);

    uint8_t* get_crf_bin(PimOpType op_type, int output_size);
#ifdef EMULATOR
    void emulator_trace_gen(unsigned int block_size, PimOpType op_type);
#endif
   private:
    cl_int exec_err_;

    pim::runtime::manager::PimManager* pim_manager_;
    pim::runtime::PimRuntime* pim_runtime_;
    std::shared_ptr<pim::runtime::manager::PimDevice> pim_device_;
    std::shared_ptr<PimCrfBinGen> pim_crf_generator_;
    PimBlockInfo* pbi_;
    PimGemvType pim_gemv_type_;
    PimKrnlType kernel_type_;
    PimGemmOrder gemm_order_;

    std::string cl_binary_path_;
    std::string cl_binary_;
    void* base_address_;

    cl_program program_;
    cl_mem d_srf_bin_buffer_;
    manager::OclBufferObj* pim_gemv_tmp_buffer_;
    cl_mem zero_buffer_;

    cl_kernel eltwise_kernel_;
    cl_kernel relu_kernel_;
    cl_kernel copy_kernel_;
    cl_kernel bn_kernel_;
    cl_kernel pim_aligned_gemm_bias_relu_8tile_fp16_;
    cl_kernel pim_aligned_gemm_bias_relu_fp16_;
    cl_kernel pim_chwise_gemm_bias_relu_32tile_fp16_;
    cl_kernel pim_chwise_gemm_bias_relu_fp16_;

#ifdef EMULATOR
    std::shared_ptr<pim::runtime::emulator::OclPimEmulator> pim_emulator_;
    PimMemTraceData* h_fmtd16_;
    PimMemTraceData* h_fmtd32_;
    PimMemTracer* d_emulator_trace_;
    cl_mem cl_d_fmtd16_;
    cl_mem cl_d_fmtd16_size_;
    cl_mem cl_d_emulator_trace_;

    size_t* h_fmtd16_size_;
    size_t* h_fmtd32_size_;
    int fmtd_size_per_ch_;
    int max_block_size_;
    int max_fmtd_size_;
#endif
};
}  // namespace executor
}  // namespace runtime
}  // namespace pim

#endif
