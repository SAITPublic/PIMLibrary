/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/ocl/OclBlockAllocator.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <list>
#include <map>
#include "executor/PimExecutor.h"
#include "manager/HostInfo.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

extern bool pim_alloc_done[MAX_NUM_GPUS];
extern uint64_t g_pim_base_addr[MAX_NUM_GPUS];
extern std::map<uint32_t, HostInfo*> host_devices;

namespace pim
{
namespace runtime
{
namespace manager
{
void* OclBlockAllocator::alloc(size_t request_size, size_t& allocated_size, int host_id) const
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    assert(request_size <= block_size() && "BlockAllocator alloc request exceeds block size.");
    uint64_t ret = 0;
    size_t bsize = block_size();

    ret = allocate_pim_block(bsize, host_id);

    if (ret == 0) return NULL;

    allocated_size = block_size();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return (void*)ret;
}

void OclBlockAllocator::free(void* ptr, size_t length) const
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

uint64_t OclBlockAllocator::allocate_pim_block(size_t bsize, int host_id) const
{
    uint64_t ret = 0;
    return ret;
}
}  // namespace manager
}  // namespace runtime
}  // namespace pim
