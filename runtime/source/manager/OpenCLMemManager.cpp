/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/OpenCLMemManager.h"
#include <CL/opencl.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "utility/assert_cl.h"
#include "utility/pim_util.h"

#ifdef __cplusplus
extern "C" {
#endif
uint64_t fmm_map_pim(uint32_t, uint32_t, uint64_t);
#ifdef __cplusplus
}
#endif

#define cl(...) cl_assert((cl##__VA_ARGS__), __FILE__, __LINE__, true);
#define cl_ok(err) cl_assert(err, __FILE__, __LINE__, true);

/*
TODO: currently we are using default device id (0) for compilation purposes.
we need to figure out how to make use of cl_device_id struct for physical id.
*/
namespace pim
{
namespace runtime
{
namespace manager
{
OpenCLMemManager::OpenCLMemManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision)
    : PimMemoryManager(pim_device, rt_type, precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

OpenCLMemManager::~OpenCLMemManager() { DLOG(INFO) << "[START] " << __FUNCTION__ << " called"; }
int OpenCLMemManager::initialize()
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int device_id_ = PimMemoryManager::initialize();
    if (device_id_ == -1) {
        return device_id_;
    }

    clGetPlatformIDs(1, &cpPlatform, NULL);
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu_devices);
    if (device_id_ != num_gpu_devices) {
        ret = -1;
        DLOG(ERROR) << "Number of GPU Ids and device count doesnt match " << __FUNCTION__ << " called";
    }

    for (int device = 0; device < num_gpu_devices; device++) {
        fragment_allocator_.push_back(new SimpleHeap<PimBlockAllocator>);
    }

    /* TODO:
        figure out what happens in case of multiple devices.
    */
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    cl_ok(err) queue = clCreateCommandQueueWithProperties(context, device_id, 0, NULL);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OpenCLMemManager::deinitialize()
{
    int ret = PimMemoryManager::deinitialize();
    return ret;
}

int OpenCLMemManager::get_physical_id()
{
    /*
    just a dummy function which reutrn a default device id.
    */
    return 2;
}

cl_context OpenCLMemManager::get_cl_context() { return context; }
cl_command_queue OpenCLMemManager::get_cl_queue() { return queue; }
cl_device_id* OpenCLMemManager::get_cl_device() { return &device_id; }
int OpenCLMemManager::alloc_memory(void** ptr, size_t size, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
        cl_ok(err);
        *ptr = (void*)device_buffer;
    } else if (mem_type == MEM_TYPE_HOST) {
        *ptr = (void*)malloc(size);
    } else if (mem_type == MEM_TYPE_PIM) {
        *ptr = fragment_allocator_[get_physical_id()]->alloc(size, get_physical_id(), rt_type_);
        cl_mem pim_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, *ptr, &err);
        cl_ok(err);
        *ptr = (void*)pim_buffer;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OpenCLMemManager::alloc_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (pim_bo->mem_type == MEM_TYPE_DEVICE) {
        cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pim_bo->size, NULL, &err);
        cl_ok(err);
        pim_bo->data = (void*)device_buffer;
    } else if (pim_bo->mem_type == MEM_TYPE_HOST) {
        /*
        makes more sense to create a host pointer of void type instead of a cl buffer.
        also aids is easy data loading into the host memory.
        */
        pim_bo->data = (void*)malloc(pim_bo->size);
    } else if (pim_bo->mem_type == MEM_TYPE_PIM) {
        pim_bo->data = fragment_allocator_[get_physical_id()]->alloc(pim_bo->size, get_physical_id(), rt_type_);
        cl_mem pim_buffer =
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, pim_bo->size, pim_bo->data, &err);
        cl_ok(err);
        pim_bo->data = (void*)pim_buffer;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OpenCLMemManager::free_memory(void* ptr, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    cl_mem curr_buff = (cl_mem)ptr;
    clFinish(queue);
    if (mem_type == MEM_TYPE_DEVICE) {
        clReleaseMemObject(curr_buff);
    }
    if (mem_type == MEM_TYPE_HOST) {
        free(ptr);
    } else if (mem_type == MEM_TYPE_PIM) {
        return fragment_allocator_[get_physical_id()]->free(ptr);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OpenCLMemManager::free_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    cl_mem curr_buff = reinterpret_cast<cl_mem>(pim_bo->data);
    err = clFinish(queue);
    cl_ok(err);
    if (pim_bo->mem_type == MEM_TYPE_DEVICE) {
        clReleaseMemObject(curr_buff);
    } else if (pim_bo->mem_type == MEM_TYPE_HOST) {
        free(pim_bo->data);
    } else if (pim_bo->mem_type == MEM_TYPE_PIM) {
        return fragment_allocator_[get_physical_id()]->free(pim_bo->data);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

// error check remaining
int OpenCLMemManager::copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    cl_mem src_buff = (cl_mem)src;
    cl_mem dst_buff = (cl_mem)dst;
    if (cpy_type == HOST_TO_PIM || cpy_type == HOST_TO_DEVICE) {
        err = clEnqueueWriteBuffer(queue, dst_buff, CL_TRUE, 0, size, src, 0, NULL, NULL);
        cl_ok(err);
    } else if (cpy_type == PIM_TO_HOST || cpy_type == DEVICE_TO_HOST) {
        err = clEnqueueReadBuffer(queue, src_buff, CL_TRUE, 0, size, dst, 0, NULL, NULL);
        cl_ok(err);
    } else if (cpy_type == DEVICE_TO_PIM || cpy_type == PIM_TO_DEVICE || cpy_type == DEVICE_TO_DEVICE) {
        err = clEnqueueCopyBuffer(queue, src_buff, dst_buff, 0, 0, size, 0, NULL, NULL);
        cl_ok(err);
    } else if (cpy_type == HOST_TO_HOST) {
        memcpy(dst, src, size);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

// error check remaining
int OpenCLMemManager::copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    size_t size = dst->size;
    cl_mem src_buff = (cl_mem)src->data;
    cl_mem dst_buff = (cl_mem)dst->data;
    if (cpy_type == HOST_TO_PIM || cpy_type == HOST_TO_DEVICE) {
        err = clEnqueueWriteBuffer(queue, dst_buff, CL_TRUE, 0, size, src->data, 0, NULL, NULL);
        cl_ok(err);
    } else if (cpy_type == PIM_TO_HOST || cpy_type == DEVICE_TO_HOST) {
        err = clEnqueueReadBuffer(queue, src_buff, CL_TRUE, 0, size, dst->data, 0, NULL, NULL);
        cl_ok(err);
    } else if (cpy_type == DEVICE_TO_PIM || cpy_type == PIM_TO_DEVICE || cpy_type == DEVICE_TO_DEVICE) {
        err = clEnqueueCopyBuffer(queue, src_buff, dst_buff, 0, 0, size, 0, NULL, NULL);
        cl_ok(err);
        err = clFinish(queue);
        cl_ok(err);
    } else if (cpy_type == HOST_TO_HOST) {
        memcpy(dst->data, src->data, size);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OpenCLMemManager::copy_memory_3d(const PimCopy3D* copy_params)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    // FIXME Need to implement

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}
}  // namespace manager
}  // namespace runtime
}  // namespace pim
