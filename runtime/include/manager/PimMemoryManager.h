/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_MEMORY_MANAGER_H_
#define _PIM_MEMORY_MANAGER_H_

#include <vector>
#include "internal/simple_heap.hpp"
#include "manager/PimBlockAllocator.h"
#include "manager/PimInfo.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
struct gpuInfo {
    uint32_t node_id;
    uint32_t gpu_id;
    uint32_t base_address;
};

class PimDevice;

class PimMemoryManager
{
   public:
    PimMemoryManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision);
    virtual ~PimMemoryManager(void);

    virtual int initialize();
    virtual int deinitialize();
    virtual int alloc_memory(void** ptr, size_t size, PimMemType mem_type) = 0;
    virtual int alloc_memory(PimBo* pim_bo) = 0;
    virtual int free_memory(void* ptr, PimMemType mem_type) = 0;
    virtual int free_memory(PimBo* pim_bo) = 0;
    virtual int copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type) = 0;
    virtual int copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type) = 0;
    virtual int convert_data_layout(void* dst, void* src, size_t size, PimOpType op_type) = 0;
    virtual int convert_data_layout(PimBo* dst, PimBo* src, PimOpType op_type) = 0;

    PimRuntimeType rt_type_;
    std::vector<SimpleHeap<PimBlockAllocator>*> fragment_allocator_;

   protected:
    PimDevice* pim_device_;
    PimPrecision precision_;
    PimBlockInfo fbi_;

    /**
     * @brief PIM Block allocator of size 2MB
     *
     * TODO: This is a simple block allocator where it uses malloc for allocation and free
     *       It has to be modified to use PIM memory region for alloc and free.
     */
};
} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_MEMORY_MANAGER_H_ */
