/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/OclMemManager.h"
#include <CL/opencl.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "utility/assert_cl.h"
#include "utility/pim_util.h"

#define cl(...) cl_assert((cl##__VA_ARGS__), __FILE__, __LINE__, true);
#define cl_ok(err_) cl_assert(err_, __FILE__, __LINE__, true);

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
OclMemManager::OclMemManager(PimDevice* pim_device, PimPrecision precision)
    : pim_device_(pim_device), precision_(precision)
{
}

OclMemManager::~OclMemManager() { DLOG(INFO) << "[START] " << __FUNCTION__ << " called"; }
int OclMemManager::initialize()
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    clGetPlatformIDs(1, &platform_, NULL);
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu_devices_);

    for (int device = 0; device < num_gpu_devices_; device++) {
        fragment_allocator_.push_back(new SimpleHeap<OclBlockAllocator>);
    }

    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_id_, NULL);
    context_ = clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err_);
    cl_ok(err_);
    queue_ = clCreateCommandQueueWithProperties(context_, device_id_, 0, NULL);

    return ret;
}

int OclMemManager::deinitialize(void)
{
    int ret = 0;
    return ret;
}

int OclMemManager::get_physical_id(void)
{
    /*
    just a dummy function which reutrn a default device id.
    */
    return 0;
}

int OclMemManager::alloc_memory(void** ptr, size_t size, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        cl_mem device_buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, NULL, &err_);
        cl_ok(err_);
        *ptr = (void*)device_buffer;
    } else if (mem_type == MEM_TYPE_HOST) {
        *ptr = (void*)malloc(size);
    } else if (mem_type == MEM_TYPE_PIM) {
        *ptr = fragment_allocator_[get_physical_id()]->alloc(size, get_physical_id());
        cl_mem pim_buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, *ptr, &err_);
        cl_ok(err_);
        *ptr = (void*)pim_buffer;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemManager::alloc_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (pim_bo->mem_type == MEM_TYPE_DEVICE) {
        cl_mem device_buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, pim_bo->size, NULL, &err_);
        cl_ok(err_);
        pim_bo->data = (void*)device_buffer;
    } else if (pim_bo->mem_type == MEM_TYPE_HOST) {
        /*
        makes more sense to create a host pointer of void type instead of a cl buffer.
        also aids is easy data loading into the host memory.
        */
        pim_bo->data = (void*)malloc(pim_bo->size);
    } else if (pim_bo->mem_type == MEM_TYPE_PIM) {
        pim_bo->data = fragment_allocator_[get_physical_id()]->alloc(pim_bo->size, get_physical_id());
        cl_mem pim_buffer =
            clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, pim_bo->size, pim_bo->data, &err_);
        cl_ok(err_);
        pim_bo->data = (void*)pim_buffer;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemManager::free_memory(void* ptr, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    cl_mem curr_buff = (cl_mem)ptr;
    clFinish(queue_);
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

int OclMemManager::free_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    cl_mem curr_buff = reinterpret_cast<cl_mem>(pim_bo->data);
    err_ = clFinish(queue_);
    cl_ok(err_);
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
int OclMemManager::copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    cl_mem src_buff = (cl_mem)src;
    cl_mem dst_buff = (cl_mem)dst;
    if (cpy_type == HOST_TO_PIM || cpy_type == HOST_TO_DEVICE) {
        err_ = clEnqueueWriteBuffer(queue_, dst_buff, CL_TRUE, 0, size, src, 0, NULL, NULL);
        cl_ok(err_);
    } else if (cpy_type == PIM_TO_HOST || cpy_type == DEVICE_TO_HOST) {
        err_ = clEnqueueReadBuffer(queue_, src_buff, CL_TRUE, 0, size, dst, 0, NULL, NULL);
        cl_ok(err_);
    } else if (cpy_type == DEVICE_TO_PIM || cpy_type == PIM_TO_DEVICE || cpy_type == DEVICE_TO_DEVICE) {
        err_ = clEnqueueCopyBuffer(queue_, src_buff, dst_buff, 0, 0, size, 0, NULL, NULL);
        cl_ok(err_);
    } else if (cpy_type == HOST_TO_HOST) {
        memcpy(dst, src, size);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

// error check remaining
int OclMemManager::copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    size_t size = dst->size;
    cl_mem src_buff = (cl_mem)src->data;
    cl_mem dst_buff = (cl_mem)dst->data;
    if (cpy_type == HOST_TO_PIM || cpy_type == HOST_TO_DEVICE) {
        err_ = clEnqueueWriteBuffer(queue_, dst_buff, CL_TRUE, 0, size, src->data, 0, NULL, NULL);
        cl_ok(err_);
    } else if (cpy_type == PIM_TO_HOST || cpy_type == DEVICE_TO_HOST) {
        err_ = clEnqueueReadBuffer(queue_, src_buff, CL_TRUE, 0, size, dst->data, 0, NULL, NULL);
        cl_ok(err_);
    } else if (cpy_type == DEVICE_TO_PIM || cpy_type == PIM_TO_DEVICE || cpy_type == DEVICE_TO_DEVICE) {
        err_ = clEnqueueCopyBuffer(queue_, src_buff, dst_buff, 0, 0, size, 0, NULL, NULL);
        cl_ok(err_);
        err_ = clFinish(queue_);
        cl_ok(err_);
    } else if (cpy_type == HOST_TO_HOST) {
        memcpy(dst->data, src->data, size);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemManager::copy_memory_3d(const PimCopy3D* copy_params)
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
