/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_EMULATOR_H_
#define _PIM_EMULATOR_H_

#include "PimTraceCoalescer.h"
#include "dramsim2/PimSimulator.h"
#include "manager/PimInfo.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace emulator
{
class PimEmulator
{
   public:
    PimEmulator();
    virtual ~PimEmulator(void) {}
    static PimEmulator* get_instance(void);

    int initialize(void);
    int deinitialize(void);
    int set_rttype(PimRuntimeType rtType)
    {
        rt_type_ = rtType;
        return 0;
    }

    int convert_mem_trace_from_16B_to_32B(PimMemTraceData* fmtd32, int* fmtd32_size, PimMemTraceData* fmtd16,
                                          int fmtd16_size, PimOpType op_type);
    int execute_bn(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size, uint64_t pim_base_addr,
                   uint8_t* temp_buf);
    int execute_elt_op(PimBo* output, PimBo* operand0, PimBo* operand1, PimMemTraceData* fmtd32, int fmtd32_size,
                       uint64_t pim_base_addr);
    int execute_relu(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size, uint64_t pim_base_addr);
    int execute_copy(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size, uint64_t pim_base_addr);
    int execute_gemm_bias_act(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                              PimOpType op_type, uint64_t pim_base_addr, uint8_t* temp_buf, PimBo* bias,
                              PimActFunc act_func);
    int execute_gemv_tile_accum(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                                PimOpType op_type, uint64_t pim_base_addr, uint8_t* temp_buf);
    int execute_gemv_add_tile_accum(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                                    PimOpType op_type, uint64_t pim_base_addr, uint8_t* temp_buf);
    int execute_gemv_tile_tree(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                               PimOpType op_type, uint64_t pim_base_addr, uint8_t* temp_buf);

   private:
    int execute_relu_bn_copy(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                             uint64_t pim_base_addr);

   private:
    PimBlockInfo fbi_;
    PimSimulator pim_sim_;

    PimRuntimeType rt_type_;
};

} /* namespace emulator */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_EMULATOR_H_ */
