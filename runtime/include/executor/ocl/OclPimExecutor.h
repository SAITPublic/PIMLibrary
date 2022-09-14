/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _OCL_PIM_EXECUTOR_H_
#define _OCL_PIM_EXECUTOR_H_

#include <CL/cl.h>
#include "emulator/PimEmulator.h"
#include "executor/IPimExecutor.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
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
    OclPimExecutor(pim::runtime::manager::PimManager* pim_manager, PimPrecision precision);
    virtual ~OclPimExecutor(void);

    int initialize(void);
    int deinitialize(void);
    int execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block);
    int execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block) { return -1; }
    int execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block) { return -1; }
    int execute_copy(PimBo* output, PimBo* pim_data, void* stream, bool block) { return -1; }
    int execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                   double epsilon, void* stream, bool block)
    {
        return -1;
    }
    int execute_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                     bool block)
    {
        return -1;
    }
    int execute_sync(void* stream) { return -1; }
    int execute_dummy(void) { return -1; }
    void* createStream(void) { return nullptr; }
   private:
    int check_cl_program_path(void);
    int build_cl_program_with_source(void);
    int save_cl_program_binary(void);
    int build_cl_program_with_binary(void);
    std::string load_cl_file(std::string filename);

   private:
    pim::runtime::manager::PimManager* pim_manager_;
    std::shared_ptr<pim::runtime::manager::PimDevice> pim_device_;
    PimPrecision precision_;
    PimBlockInfo* pbi_;
    int max_crf_size_;
    std::string cl_binary_path_;
    std::string cl_binary_;

    cl_program program_;
    cl_mem d_srf_bin_buffer_;
    cl_mem pim_gemv_tmp_buffer_;
    cl_mem zero_buffer_;
};
}  // namespace executor
}  // namespace runtime
}  // namespace pim

#endif