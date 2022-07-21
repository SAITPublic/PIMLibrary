/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _OPENCL_MEM_MANAGER_H_
#define _OPENCL_MEM_MANAGER_H_

#include <CL/cl.h>
#include "internal/simple_heap.hpp"
#include "manager/PimInfo.h"
#include "manager/PimMemoryManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class OpenCLMemManager : public PimMemoryManager
{
   public:
    OpenCLMemManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision);
    virtual ~OpenCLMemManager();

    int initialize();
    int deinitialize();
    int alloc_memory(void** ptr, size_t size, PimMemType mem_type);
    int alloc_memory(PimBo* pim_bo);
    int free_memory(void* ptr, PimMemType mem_type);
    int free_memory(PimBo* pim_bo);
    int copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type);
    int copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type);
    int copy_memory_3d(const PimCopy3D* copy_params);
    int get_physical_id();
    cl_context get_cl_context();
    cl_command_queue get_cl_queue();
    cl_device_id* get_cl_device();
    cl_command_queue queue;  // command queue
    cl_context context;      // context
    cl_device_id device_id;  // device ID

   private:
    cl_platform_id cpPlatform;  // OpenCL platform
    cl_uint num_gpu_devices;    // num gpu devices
    cl_int err;
};
}  // namespace manager
}  // namespace runtime
}  // namespace pim

#endif /*_OPENCL_MEM_MANAGER_H_ */
