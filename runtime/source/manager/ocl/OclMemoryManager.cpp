/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/ocl/OclMemoryManager.h"
#include <CL/opencl.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "manager/HostInfo.h"
#include "utility/assert_cl.h"
#include "utility/pim_util.h"

/*
TODO: currently we are using default device id (0) for compilation purposes.
we need to figure out how to make use of cl_device_id struct for physical id.
*/
extern uint64_t g_pim_base_addr[MAX_NUM_GPUS];

namespace pim
{
namespace runtime
{
cl_platform_id platform;
cl_context context;
cl_device_id device_id;
cl_command_queue queue;

namespace manager
{
OclMemoryManager::OclMemoryManager(std::shared_ptr<PimDevice> pim_device, PimPrecision precision)
    : pim_device_(pim_device)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu_devices_);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device_id, 0, NULL);

    for (int device = 0; device < num_gpu_devices_; device++) {
        fragment_allocator_.push_back(std::make_shared<SimpleHeap<OclBlockAllocator>>());
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

OclMemoryManager::~OclMemoryManager()
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    pim_device_.reset();
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    fragment_allocator_.clear();
}

int OclMemoryManager::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    return ret;
}

int OclMemoryManager::deinitialize(void)
{
    int ret = 0;

    return ret;
}

int OclMemoryManager::get_physical_id(void)
{
    /*
    just a dummy function which reutrn a default device id.
    */
    return 0;
}

int OclMemoryManager::alloc_memory(void** ptr, size_t size, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int err = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
        cl_ok(err);
        *ptr = (void*)device_buffer;
    } else if (mem_type == MEM_TYPE_HOST) {
        *ptr = (void*)malloc(size);
    } else if (mem_type == MEM_TYPE_PIM) {
        // it uses fragment allocator to :
        // 1. either allocates a pim block of block size if pim_alloc_done is false and returns the base address as the
        // requested buffer address.
        // 2. else return a virtual address of the buffer in the above allocated region which has to be then allocted as
        // Subbuffer of above buffer in opencl.
        // Note: get_cl_mem_obj return a base buffer in device address space and fragment allocator returns host mapped
        // address.
        void* local_buffer = fragment_allocator_[get_physical_id()]->alloc(size, get_physical_id());

        // create sub buffer object in host mapped device addres space.
        cl_mem base_buffer = (cl_mem)fragment_allocator_[0]->get_pim_base();
        cl_buffer_region sub_buffer_region = {(uint64_t)local_buffer - g_pim_base_addr[get_physical_id()], size};
        cl_mem sub_buffer = clCreateSubBuffer(base_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                              (void*)&sub_buffer_region, &err);
        cl_ok(err);

        OclBufferObj* buffer = new OclBufferObj;
        buffer->host_addr = (uint64_t)local_buffer;
        buffer->dev_addr = sub_buffer;
        buffer->size = size;

        *ptr = (void*)buffer;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemoryManager::alloc_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int err = 0;

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
        // it uses fragment allocator to :
        // 1. either allocates a pim block of block size if pim_alloc_done is false and returns the base address as the
        // requested buffer address.
        // 2. else return a virtual address of the buffer in the above allocated region which has to be then allocted as
        // Subbuffer of above buffer in opencl.
        // Note: get_cl_mem_obj return a base buffer in device address space and fragment allocator returns host mapped
        // address.
        void* local_buffer = fragment_allocator_[get_physical_id()]->alloc(pim_bo->size, get_physical_id());

        // create sub buffer object in host mapped device addres space.
        cl_mem base_buffer = (cl_mem)fragment_allocator_[0]->get_pim_base();
        cl_buffer_region sub_buffer_region = {(uint64_t)local_buffer - g_pim_base_addr[get_physical_id()],
                                              pim_bo->size};
        cl_mem sub_buffer = clCreateSubBuffer(base_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                              (void*)&sub_buffer_region, &err);
        cl_ok(err);

        OclBufferObj* buffer = new OclBufferObj;
        buffer->host_addr = (uint64_t)local_buffer;
        buffer->dev_addr = sub_buffer;
        buffer->size = pim_bo->size;

        pim_bo->data = (void*)buffer;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemoryManager::free_memory(void* ptr, PimMemType mem_type)
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
        OclBufferObj* buffer = (OclBufferObj*)ptr;

        fragment_allocator_[get_physical_id()]->free((void*)buffer->host_addr);
        cl_mem cl_buffer_obj = (cl_mem)buffer->dev_addr;

        clReleaseMemObject(cl_buffer_obj);
        free(buffer);
        ptr = nullptr;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemoryManager::free_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int err = 0;

    cl_mem curr_buff = reinterpret_cast<cl_mem>(pim_bo->data);
    err = clFinish(queue);
    cl_ok(err);
    if (pim_bo->mem_type == MEM_TYPE_DEVICE) {
        clReleaseMemObject(curr_buff);
    } else if (pim_bo->mem_type == MEM_TYPE_HOST) {
        free(pim_bo->data);
    } else if (pim_bo->mem_type == MEM_TYPE_PIM) {
        OclBufferObj* buffer = (OclBufferObj*)pim_bo->data;

        fragment_allocator_[get_physical_id()]->free((void*)buffer->host_addr);
        cl_mem cl_buffer_obj = (cl_mem)buffer->dev_addr;

        clReleaseMemObject(cl_buffer_obj);
        free(buffer);
        pim_bo->data = nullptr;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

// error check remaining
int OclMemoryManager::copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int err = 0;

    cl_mem src_buff = (cl_mem)src;
    cl_mem dst_buff = (cl_mem)dst;

    uint64_t src_addr, dst_addr;
    switch (cpy_type) {
        case HOST_TO_PIM:
            src_addr = (uint64_t)src;
            dst_addr = ((OclBufferObj*)dst)->host_addr;
            memcpy((void*)dst_addr, (void*)src_addr, size);
            break;
        case HOST_TO_DEVICE:
            err = clEnqueueWriteBuffer(queue, dst_buff, CL_TRUE, 0, size, src, 0, NULL, NULL);
            cl_ok(err);
            break;
        case PIM_TO_HOST:
            src_addr = ((OclBufferObj*)src)->host_addr;
            dst_addr = (uint64_t)dst;
            memcpy((void*)dst_addr, (void*)src_addr, size);
            break;
        case DEVICE_TO_HOST:
            err = clEnqueueReadBuffer(queue, src_buff, CL_TRUE, 0, size, dst, 0, NULL, NULL);
            cl_ok(err);
            break;
        case DEVICE_TO_PIM:
            err = clEnqueueReadBuffer(queue, src_buff, CL_TRUE, 0, size, (void*)((OclBufferObj*)dst_buff)->host_addr, 0,
                                      NULL, NULL);
            cl_ok(err);
            break;
        case PIM_TO_DEVICE:
            err = clEnqueueWriteBuffer(queue, dst_buff, CL_TRUE, 0, size, (void*)((OclBufferObj*)src_buff)->host_addr,
                                       0, NULL, NULL);
            cl_ok(err);
            break;
        case DEVICE_TO_DEVICE:
            err = clEnqueueCopyBuffer(queue, src_buff, dst_buff, 0, 0, size, 0, NULL, NULL);
            cl_ok(err);
            err = clFinish(queue);
            cl_ok(err);
            break;
        case HOST_TO_HOST:
            memcpy(dst, src, size);
            break;
        default:
            DLOG(ERROR) << "Invalid copy type";
            break;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

// error check remaining
int OclMemoryManager::copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int err = 0;

    size_t size = dst->size;
    uint64_t src_addr, dst_addr;

    cl_mem src_buff = (cl_mem)src->data;
    cl_mem dst_buff = (cl_mem)dst->data;

    switch (cpy_type) {
        case HOST_TO_PIM:
            src_addr = (uint64_t)src->data;
            dst_addr = ((OclBufferObj*)dst->data)->host_addr;
            memcpy((void*)dst_addr, (void*)src_addr, size);
            break;
        case HOST_TO_DEVICE:
            err = clEnqueueWriteBuffer(queue, dst_buff, CL_TRUE, 0, size, src->data, 0, NULL, NULL);
            cl_ok(err);
            break;
        case PIM_TO_HOST:
            src_addr = ((OclBufferObj*)src->data)->host_addr;
            dst_addr = (uint64_t)dst->data;
            memcpy((void*)dst_addr, (void*)src_addr, size);
            break;
        case DEVICE_TO_HOST:
            err = clEnqueueReadBuffer(queue, src_buff, CL_TRUE, 0, size, dst->data, 0, NULL, NULL);
            cl_ok(err);
            break;
        case DEVICE_TO_PIM:
            err = clEnqueueReadBuffer(queue, src_buff, CL_TRUE, 0, size, (void*)((OclBufferObj*)dst_buff)->host_addr, 0,
                                      NULL, NULL);
            cl_ok(err);
            break;
        case PIM_TO_DEVICE:
            err = clEnqueueWriteBuffer(queue, dst_buff, CL_TRUE, 0, size, (void*)((OclBufferObj*)src_buff)->host_addr,
                                       0, NULL, NULL);
            cl_ok(err);
            break;
        case DEVICE_TO_DEVICE:
            err = clEnqueueCopyBuffer(queue, src_buff, dst_buff, 0, 0, size, 0, NULL, NULL);
            cl_ok(err);
            err = clFinish(queue);
            cl_ok(err);
            break;
        case HOST_TO_HOST:
            memcpy(dst->data, src->data, size);
            break;
        default:
            DLOG(ERROR) << "Invalid copy type";
            break;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemoryManager::copy_memory_3d(const PimCopy3D* copy_params)
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
