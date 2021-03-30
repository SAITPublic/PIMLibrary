/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_MEMORY_MANAGER_H_
#define _FIM_MEMORY_MANAGER_H_

#include "fim_data_types.h"
#include "internal/simple_heap.hpp"
#include "manager/FimInfo.h"
#include "manager/FimManager.h"

namespace fim
{
namespace runtime
{
namespace manager
{
class FimDevice;

class FimMemoryManager
{
   public:
    FimMemoryManager(FimDevice* fim_device, FimRuntimeType rt_type, FimPrecision precision);
    virtual ~FimMemoryManager(void);

    int initialize(void);
    int deinitialize(void);
    int alloc_memory(void** ptr, size_t size, FimMemType mem_type);
    int alloc_memory(FimBo* fim_bo);
    int free_memory(void* ptr, FimMemType mem_type);
    int free_memory(FimBo* fim_bo);
    int copy_memory(void* dst, void* src, size_t size, FimMemCpyType cpy_type);
    int copy_memory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type);
    int convert_data_layout(void* dst, void* src, size_t size, FimOpType op_type);
    int convert_data_layout(FimBo* dst, FimBo* src, FimOpType op_type);
    int convert_data_layout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type);

   private:
    int convert_data_layout_for_elt_op(FimBo* dst, FimBo* src, FimBankType fim_bank_type);
    int convert_data_layout_for_gemv_weight(FimBo* dst, FimBo* src);
    int convert_data_layout_for_relu(FimBo* dst, FimBo* src);
    int convert_data_layout_for_bn(FimBo* dst, FimBo* src);

   private:
    FimDevice* fim_device_;
    FimRuntimeType rt_type_;
    FimPrecision precision_;
    FimBlockInfo fbi_;

    /**
     * @brief FIM Block allocator of size 2MB
     *
     * TODO: This is a simple block allocator where it uses malloc for allocation and free
     *       It has to be modified to use FIM memory region for alloc and free.
     */

    class FimBlockAllocator
    {
       private:
#if RADEON7
        static const size_t block_size_ = 8589934592;  // 8GB Fim area
#else
        static const size_t block_size_ = 17179869184;  // 16GB Fim area
#endif
       public:
        explicit FimBlockAllocator() {}
        void* alloc(size_t request_size, size_t& allocated_size) const;
        void free(void* ptr, size_t length) const;
        uint64_t allocate_fim_block(size_t request_size) const;
        size_t block_size() const { return block_size_; }
    };

    SimpleHeap<FimBlockAllocator> fragment_allocator_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_MEMORY_MANAGER_H_ */
