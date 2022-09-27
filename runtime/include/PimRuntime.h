/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_RUNTIME_H_
#define _PIM_RUNTIME_H_

#pragma GCC diagnostic ignored "-Wunused-private-field"

#include <unordered_map>
#include "executor/IPimExecutor.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
class PimRuntime
{
   public:
    PimRuntime(PimRuntimeType rtType, PimPrecision precision);
    virtual ~PimRuntime(void);

    int initialize(void);
    int deinitialize(void);
    int set_device(uint32_t device_id);
    int get_device(uint32_t* device_id);
    void* createStream(PimRuntimeType rt_type);
    int alloc_memory(void** ptr, size_t size, PimMemType mem_type);
    int alloc_memory(PimBo* pimBo, void* user_ptr = nullptr);
    int free_memory(void* ptr, PimMemType mem_type);
    int free_memory(PimBo* pimBo);
    int copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type);
    int copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type);
    int copy_memory_3d(const PimCopy3D* copy_params);

    int execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block = false);
    int execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block = false);
    int execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block = false);
    int execute_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                     bool block);
    int execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                   double epsilon, void* stream, bool block = false);
    int execute_sync(void* stream);
    int execute_dummy(void);
    PimBo* generate_gemm_weight_from_buffer(PimBo* src, bool save_for_reuse = false);
    PimBo* get_preloaded_pim_gemm_weight(PimBo* dev_wei, bool save_for_reuse = true);

   private:
    int insert_preloaded_pim_weight(PimBo* dev_wei, PimBo* pim_wei);
    PimBo* find_preloaded_pim_weight(PimBo* dev_wei);
    pim::runtime::manager::PimManager* pim_manager_;
    std::shared_ptr<pim::runtime::manager::PimDevice> pim_device_;
    std::shared_ptr<executor::IPimExecutor> pim_executor_;
    PimRuntimeType rt_type_;
    PimPrecision precision_;
    std::unordered_map<uint32_t, PimBo*> weight_map_;
};

} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_RUNTIME_H_ */
