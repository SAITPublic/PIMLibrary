/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_BLOCK_ALLOCATOR_H_
#define _PIM_BLOCK_ALLOCATOR_H_

#include "manager/PimInfo.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
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
    void* alloc(size_t request_size, size_t& allocated_size, int device_id, PimRuntimeType rt_type) const;
    void free(void* ptr, size_t length) const;
    uint64_t allocate_pim_block(size_t request_size, int device_id, PimRuntimeType rt_type) const;
    size_t block_size() const { return block_size_; }
};
}
}
}

#endif /*_PIM_BLOCK_ALLOCATOR_H_ */
