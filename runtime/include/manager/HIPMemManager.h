/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _HIP_MEM_MANAGER_H_
#define _HIP_MEM_MANAGER_H_

#include <vector>
#include "internal/simple_heap.hpp"
#include "manager/PimInfo.h"
#include "manager/PimMemoryManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class HIPMemManager : public PimMemoryManager
{
   public:
    HIPMemManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision);
    virtual ~HIPMemManager();

    int initialize();
    int deinitialize();
    int alloc_memory(void** ptr, size_t size, PimMemType mem_type);
    int alloc_memory(PimBo* pim_bo);
    int free_memory(void* ptr, PimMemType mem_type);
    int free_memory(PimBo* pim_bo);
    int copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type);
    int copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type);
    int convert_data_layout(void* dst, void* src, size_t size, PimOpType op_type);
    int convert_data_layout(PimBo* dst, PimBo* src, PimOpType op_type);

   private:
    int convert_data_layout_for_gemv_weight(PimBo* dst, PimBo* src);
    int num_gpu_devices_;
    int device_id;
};
}
}
}

#endif /*_HIP_MEM_MANAGER_H_*/
