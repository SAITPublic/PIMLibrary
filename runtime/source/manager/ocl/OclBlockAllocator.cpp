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
#include "manager/HostInfo.h"
#include "utility/assert_cl.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

extern bool pim_alloc_done[MAX_NUM_GPUS];
extern uint64_t g_pim_base_addr[MAX_NUM_GPUS];
extern std::map<uint32_t, HostInfo*> host_devices;

namespace pim
{
namespace runtime
{
extern cl_context context;
extern cl_command_queue queue;

namespace manager
{
void* OclBlockAllocator::alloc(size_t request_size, size_t& allocated_size, int host_id)
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

void OclBlockAllocator::free(void* ptr, size_t length)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    cl_int cl_err;
    clEnqueueUnmapMemObject(queue, base_address_memobject_, base_host_address_, NULL, NULL, NULL);
    cl_err = clReleaseMemObject(base_address_memobject_);
    cl_ok(cl_err);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

uint64_t OclBlockAllocator::allocate_pim_block(size_t bsize, int host_id)
{
    uint64_t ret = 0;
    DLOG(INFO) << "Device ID : " << host_id;
    if (pim_alloc_done[host_id] == true) return 0;

    base_address_memobject_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bsize, NULL, NULL);
    base_host_address_ = clEnqueueMapBuffer(queue, base_address_memobject_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                            bsize, 0, NULL, NULL, NULL);
    ret = (uint64_t)base_host_address_;

    if (ret) {
        pim_alloc_done[host_id] = true;
        g_pim_base_addr[host_id] = ret;
    } else {
        std::cout << "fmm_map_pim failed! " << ret << std::endl;
    }

    return ret;
}
}  // namespace manager
}  // namespace runtime
}  // namespace pim
