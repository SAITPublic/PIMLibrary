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

#ifndef _OCL_BLOCK_ALLOCATOR_H_
#define _OCL_BLOCK_ALLOCATOR_H_

#include "CL/cl.h"
#include "manager/PimInfo.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class OclBlockAllocator
{
    /**
     * @brief OCL Block allocator of size 2MB
     *
     * TODO: This is a simple block allocator where it uses malloc for allocation and free
     *       It has to be modified to use PIM memory region for alloc and free.
     */

   public:
    explicit OclBlockAllocator(void) {}
    void* alloc(size_t request_size, size_t& allocated_size, int host_id);
    void free(void* ptr, size_t length);
    uint64_t allocate_pim_block(size_t request_size, int host_id);
    size_t block_size(void) const { return block_size_; }
    void* get_pim_base() { return (void*)base_address_memobject_; };

   private:
    cl_mem base_address_memobject_;
    void* base_host_address_;
    // static const size_t block_size_ = 134217728;  // 128M Pim area

    // target mode space.
    static const size_t block_size_ = 1073741824;  // 1GB PIM area
};

}  // namespace manager
}  // namespace runtime
}  // namespace pim

#endif /*_OCL_BLOCK_ALLOCATOR_H_ */
