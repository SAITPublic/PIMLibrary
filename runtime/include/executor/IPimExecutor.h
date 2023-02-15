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

#ifndef _IPIM_EXECUTOR_H_
#define _IPIM_EXECUTOR_H_

#include "manager/PimInfo.h"
#include "manager/PimManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace executor
{
class IPimExecutor
{
   public:
    virtual int initialize(void) = 0;
    virtual int deinitialize(void) = 0;
    virtual int execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block) = 0;
    virtual int execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block) = 0;
    virtual int execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block) = 0;
    virtual int execute_copy(PimBo* output, PimBo* pim_data, void* stream, bool block) = 0;
    virtual int execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                           double epsilon, void* stream, bool block) = 0;
    virtual int execute_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                             bool block) = 0;
    virtual int execute_custom_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, bool is_gemv_add, void* stream,
                                    bool block) = 0;
    virtual int execute_custom_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, PimBo* operand2, bool relu,
                                        void* stream, bool block) = 0;
    virtual int execute_sync(void* stream) = 0;
    virtual int execute_dummy(void) = 0;
    virtual void* create_stream(void) = 0;
    virtual void set_gemm_order(PimGemmOrder gemm_order) = 0;
    virtual int set_device(uint32_t device_id) = 0;
    virtual int get_device(uint32_t *device_id) = 0;
};

} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */

#endif /* _IPIM_EXECUTOR_H_ */
