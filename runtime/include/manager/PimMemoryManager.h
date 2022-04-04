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

#pragma GCC diagnostic ignored "-Wunused-private-field"
#include <vector>

#include "internal/simple_heap.hpp"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
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

class PimBlockAllocator
{
   private:
#if EMULATOR
    static const size_t block_size_ = 134217728; // 128M Pim area
#elif RADEON7
    static const size_t block_size_ = 8589934592;  // 8GB Pim area
#else
    static const size_t block_size_ = 17179869184;  // 16GB Pim area
#endif
   public:
    explicit PimBlockAllocator() {}
    void* alloc(size_t request_size, size_t& allocated_size) const;
    void free(void* ptr, size_t length) const;
    uint64_t allocate_pim_block(size_t request_size) const;
    size_t block_size() const { return block_size_; }
};

class PimMemoryManager
{
   public:
    PimMemoryManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision);
    virtual ~PimMemoryManager(void);

    int initialize(void);
    int deinitialize(void);
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

   private:
    PimDevice* pim_device_;
    PimRuntimeType rt_type_;
    PimPrecision precision_;
    PimBlockInfo fbi_;
    int num_gpu_devices_;

    /**
     * @brief PIM Block allocator of size 2MB
     *
     * TODO: This is a simple block allocator where it uses malloc for allocation and free
     *       It has to be modified to use PIM memory region for alloc and free.
     */

    std::vector<SimpleHeap<PimBlockAllocator>*> fragment_allocator_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_MEMORY_MANAGER_H_ */
