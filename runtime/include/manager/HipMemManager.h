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

#include <map>
#include <vector>
#include "internal/simple_heap.hpp"
#include "manager/HipBlockAllocator.h"
#include "manager/HostInfo.h"
#include "manager/IPimMemoryManager.h"
#include "manager/PimDevice.h"
#include "manager/PimInfo.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class PimDevice;
class HipMemManager : public IPimMemoryManager
{
   public:
    HipMemManager(PimDevice* pim_device, PimPrecision precision);
    virtual ~HipMemManager(void);

    int initialize(void);
    int deinitialize(void);
    int alloc_memory(void** ptr, size_t size, PimMemType mem_type);
    int alloc_memory(PimBo* pim_bo);
    int free_memory(void* ptr, PimMemType mem_type);
    int free_memory(PimBo* pim_bo);
    int copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type);
    int copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type);
    int copy_memory_3d(const PimCopy3D* copy_params);
    int convert_data_layout(PimBo* dst, PimBo* src, PimOpType op_type);

    void* get_context(void) { return nullptr; }
    void* get_queue(void) { return nullptr; }
    void* get_device(void) { return nullptr; }

   private:
    int convert_data_layout_for_gemm_weight(PimBo* dst, PimBo* src);
    int convert_data_layout_for_aligned_gemm_weight(PimBo* dst, PimBo* src);
    int convert_data_layout_for_chwise_gemm_weight(PimBo* dst, PimBo* src);
    int convert_data_layout_for_gemv_weight(PimBo* dst, PimBo* src, int data_offset);
    int convert_data_layout_for_gemv_weight(PimBo* dst, PimBo* src, int data_offset, int ch_per_op);

   private:
    std::vector<SimpleHeap<HipBlockAllocator>*> fragment_allocator_;
    int num_gpu_devices_;
    int host_id_;
    PimDevice* pim_device_;
    PimPrecision precision_;
    PimBlockInfo fbi_;
};
}  // namespace manager
}  // namespace runtime
}  // namespace pim

#endif /*_HIP_MEM_MANAGER_H_*/
