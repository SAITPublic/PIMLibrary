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
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <list>
#include "executor/PimExecutor.h"
#include "manager/PimBlockAllocator.h"
#include "manager/HostInfo.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

std::map<uint32_t, HostInfo*> host_devices;

namespace pim
{
namespace runtime
{
namespace manager
{
inline std::list<int> get_env(const char* key)
{
    std::list<int> hip_devices = {};
    if (key == nullptr) {
        return hip_devices;
    }

    if (*key == '\0') {
        return hip_devices;
    }

    const char* ev_val = getenv(key);
    if (ev_val == nullptr) {
        return hip_devices;  // variable not defined
    }

    std::string env = getenv(key);
    std::string delimiter = ",";
    size_t pos = 0;
    std::string token;
    while ((pos = env.find(delimiter)) != std::string::npos) {
        token = env.substr(0, pos);
        int num = stoi((token));
        hip_devices.push_back(num);
        env.erase(0, pos + delimiter.length());
    }
    int num = stoi((env));
    hip_devices.push_back(num);

    return hip_devices;
}

PimMemoryManager::PimMemoryManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision)
    : rt_type_(rt_type), pim_device_(pim_device), precision_(precision)
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
    int host_id = 0;
    int num_gpu_devices = 0;
    std::list<int> hip_visible_devices = get_env("HIP_VISIBLE_DEVICES");
    hipGetDeviceCount(&num_gpu_devices);

    // if hip_device is not set , then assume all devices are visible
    if (hip_visible_devices.empty()) {
        for (int device = 0; device < num_gpu_devices; device++) {
            hip_visible_devices.push_back(device);
        }
    }

    int curr = 0;
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
        if (gpu_id == 0) continue;
        if (gpu_id != 0 && curr == hip_visible_devices.front()) {
            DLOG(INFO) << " adding device:" << id << " "
                       << "gpu_id:" << gpu_id;
            HostInfo* host_info = new HostInfo;
            host_info->host_type = AMDGPU;
            host_info->node_id = id;
            host_info->host_id = gpu_id;
            host_info->base_address = 0;
            host_devices[host_id] = host_info;
            host_id++;
            hip_visible_devices.pop_front();
        }
        curr++;
    }

    if (host_id == 0) {
        ret = -1;
        DLOG(ERROR) << "GPU device not found " << __FUNCTION__ << " called";
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return host_id;
}

int PimMemoryManager::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */
