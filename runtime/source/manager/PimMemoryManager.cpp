/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/PimMemoryManager.h"
#include "executor/PimExecutor.h"
#include "manager/PimBlockAllocator.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifndef EMULATOR
uint64_t fmm_map_pim(uint32_t, uint32_t, uint64_t);
#endif
#ifdef __cplusplus
}
#endif

extern bool pim_alloc_done[MAX_NUM_GPUS];
extern uint64_t g_pim_base_addr[MAX_NUM_GPUS];
namespace pim
{
namespace runtime
{
namespace manager
{
std::map<uint32_t, gpuInfo*> gpu_devices;
PimMemoryManager::PimMemoryManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision)
    : pim_device_(pim_device), rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    get_pim_block_info(&fbi_);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

PimMemoryManager::~PimMemoryManager() { DLOG(INFO) << "[START] " << __FUNCTION__ << " called"; }
int PimMemoryManager::initialize()
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int max_topology = 32;
    FILE* fd;
    char path[256];
    uint32_t gpu_id;
    int device_id = 0;

    for (int id = 0; id < max_topology; id++) {
        // Get GPU ID
        snprintf(path, 256, "/sys/devices/virtual/kfd/kfd/topology/nodes/%d/gpu_id", id);
        fd = fopen(path, "r");
        if (!fd) continue;
        if (fscanf(fd, "%ul", &gpu_id) != 1) {
            fclose(fd);
            continue;
        }

        fclose(fd);
        if (gpu_id != 0) {
            gpuInfo* device_info = new gpuInfo;
            device_info->node_id = id;
            device_info->gpu_id = gpu_id;
            device_info->base_address = 0;
            gpu_devices[device_id] = device_info;
            device_id++;
        }
    }

    if (device_id == 0) {
        ret = -1;
        DLOG(ERROR) << "GPU device not found " << __FUNCTION__ << " called";
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return device_id;
}

int PimMemoryManager::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

void* PimBlockAllocator::alloc(size_t request_size, size_t& allocated_size, int device_id, PimRuntimeType rt_type) const
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    assert(request_size <= block_size() && "BlockAllocator alloc request exceeds block size.");
    uint64_t ret = 0;
    size_t bsize = block_size();

    ret = allocate_pim_block(bsize, device_id, rt_type);

    if (ret == 0) return NULL;

    allocated_size = block_size();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return (void*)ret;
}

void PimBlockAllocator::free(void* ptr, size_t length) const
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    if (ptr == NULL || length == 0) {
        return;
    }

    /* todo:implement pimfree function */
    std::free(ptr);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

uint64_t PimBlockAllocator::allocate_pim_block(size_t bsize, int device_id, PimRuntimeType rt_type) const
{
    uint64_t ret = 0;
    std::cout << "Device ID :" << device_id << std::endl;
    if (pim_alloc_done[device_id] == true) return 0;

#ifdef EMULATOR
    if (hipMalloc((void**)&ret, bsize) != hipSuccess) {
        std::cout << "fmm_map_pim failed! " << ret << std::endl;
        return -1;
    }
#else
    ret = fmm_map_pim(gpu_devices[device_id]->node_id, gpu_devices[device_id]->gpu_id, bsize);
#endif
    if (ret) {
        pim_alloc_done[device_id] = true;
        g_pim_base_addr[device_id] = ret;
#ifndef EMULATOR
        if (rt_type == RT_TYPE_HIP) {
            hipHostRegister((void*)g_pim_base_addr[device_id], bsize, hipRegisterExternalSvm);
        }
#endif
    } else {
        std::cout << "fmm_map_pim failed! " << ret << std::endl;
    }
    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */
