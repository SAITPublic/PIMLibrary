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

#include "manager/ocl/OclMemoryManager.h"
#include <CL/opencl.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "half.hpp"
#include "manager/HostInfo.h"
#include "utility/assert_cl.h"
#include "utility/pim_util.h"
using half_float::half;
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
    pbi_ = pim_device_->get_pim_block_info();

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

int OclMemoryManager::convert_data_layout(PimBo* dst, PimBo* src, bool reorder_on_device, void* stream)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    bool is_chwise = check_chwise_gemm_bo(src, gemm_order_);
    // TODO: support device reordering for OCL
    if (is_chwise) {
        ret = convert_data_layout_for_chwise_gemm_weight(dst, src);
    } else {
        ret = convert_data_layout_for_aligned_gemm_weight(dst, src);
    }
    if (ret != 0) {
        printf("fail to convert data layout for gemm\n");
        return ret;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemoryManager::convert_data_layout_for_chwise_gemm_weight(PimBo* dst, PimBo* src)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    int num_grf_A = pbi_->num_grf;
    int num_grf_B = pbi_->num_grf;
    int num_pim_blocks = pbi_->num_pim_blocks;
    int num_pim_chan = pbi_->num_pim_chan;
    int num_pim_rank = pbi_->num_pim_rank;
    int num_banks = pbi_->num_banks;
    int num_bank_groups = pbi_->num_bank_groups;
    int trans_size = pbi_->trans_size;

    char* dst_data = nullptr;
    char* src_data = nullptr;

    int cidx = 0;
    int rank = 0;
    int bg = 0;
    int bank = 0;
    uint32_t col = 0;
    uint32_t row = 0;
    uint64_t addr = 0;
    uint32_t even_s_row = 0;  // starting_row;
    uint32_t even_s_col = 0;  // starting_col;
    uint32_t odd_s_row = 0;   // starting_row;
    uint32_t odd_s_col = 0;   // starting_col;

    int type_size = (src->precision == PIM_FP16) ? 2 : 1;

    int in_tile_size = num_grf_A;
    int out_tile_size = num_grf_B * num_pim_blocks * num_pim_chan * num_pim_rank;
    int data_offset = 0;

    int iter_cnt = 0;
    int in_cnt = 0;

    if (gemm_order_ == I_X_W) {
        iter_cnt = src->bshape.n * src->bshape.c * src->bshape.w / PIM_GEMV_OUT_ALIGN;
        in_cnt = src->bshape.h * type_size / trans_size;
    } else {
        iter_cnt = src->bshape.n * src->bshape.c * src->bshape.h / PIM_GEMV_OUT_ALIGN;
        in_cnt = src->bshape.w * type_size / trans_size;
    }

    for (int iter = 0; iter < iter_cnt; iter++) {
        cidx = 0;
        rank = 0;
        bg = 0;
        bank = 0;
        col = 0;
        row = 0;
        addr = 0;
        even_s_row = 0;
        even_s_col = 0;
        odd_s_row = 0;
        odd_s_col = 0;
        dst_data = (char*)dst->data + data_offset;
        src_data = (char*)src->data + data_offset;

        for (int x = 0; x < in_cnt; x += in_tile_size) {
            if ((x / in_tile_size) % 2 == 0) {
                for (int tiled_y = 0; tiled_y < out_tile_size; tiled_y += num_grf_B) {
                    col = even_s_col;
                    row = even_s_row;

                    for (int grfb_idx = 0; grfb_idx < num_grf_B; grfb_idx++) {
                        for (int grfa_idx = 0; grfa_idx < num_grf_A; grfa_idx++) {
                            addr = addr_gen_safe(cidx, rank, bg, bank, row, col);
#ifdef EMULATOR
                            int d_idx = (tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
#else
                            int d_idx = (tiled_y + grfb_idx) * in_cnt + x + grfa_idx;
#endif
                            memcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size);
                            col++;
                        }
                    }

                    bank += (num_banks / num_pim_blocks);

                    if (bank >= (num_banks / num_bank_groups)) {
                        bg++;
                        bank = 0;
                    }

                    if (bg >= num_bank_groups) {
                        bg = 0;
                        rank++;
                    }

                    if (rank >= num_pim_rank) {
                        rank = 0;
                        cidx++;
                    }

                    if (cidx >= num_pim_chan) {
                        cidx = 0;
                        even_s_row = row;
                        even_s_col = col;
                    }
                }
            } else if ((x / in_tile_size) % 2 == 1) {
                for (int tiled_y = 0; tiled_y < out_tile_size; tiled_y += num_grf_B) {
                    col = odd_s_col;
                    row = odd_s_row;

                    for (int grfb_idx = 0; grfb_idx < num_grf_B; grfb_idx++) {
                        for (int grfa_idx = 0; grfa_idx < num_grf_A; grfa_idx++) {
                            addr = addr_gen_safe(cidx, rank, bg, bank + 1, row, col);
#ifdef EMULATOR
                            int d_idx = (tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
#else
                            int d_idx = (tiled_y + grfb_idx) * in_cnt + x + grfa_idx;
#endif
                            memcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size);
                            col++;
                        }
                    }

                    bank += (num_banks / num_pim_blocks);

                    if (bank >= (num_banks / num_bank_groups)) {
                        bg++;
                        bank = 0;
                    }

                    if (bg >= num_bank_groups) {
                        bg = 0;
                        rank++;
                    }

                    if (rank >= num_pim_rank) {
                        rank = 0;
                        cidx++;
                    }

                    if (cidx >= num_pim_chan) {
                        cidx = 0;
                        odd_s_row = row;
                        odd_s_col = col;
                    }
                }
            }
        }
        data_offset += (src->bshape.h * PIM_GEMV_OUT_ALIGN * sizeof(half_float::half));
    }
    dst->data_layout_type = PimDataLayoutType::CHWISE_GEMM_WEIGHT;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclMemoryManager::convert_data_layout_for_aligned_gemm_weight(PimBo* dst, PimBo* src)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int num_grf_A = pbi_->num_grf;
    int num_grf_B = pbi_->num_grf;
    int num_pim_blocks = pbi_->num_pim_blocks;
    int num_pim_chan = pbi_->num_pim_chan;
    int num_pim_rank = pbi_->num_pim_rank;
    int num_banks = pbi_->num_banks;
    int num_bank_groups = pbi_->num_bank_groups;
    int trans_size = pbi_->trans_size;

    int in_tile_size = num_grf_A;
    int out_tile_size = num_grf_B * num_pim_blocks * num_pim_chan * num_pim_rank;
    char* dst_data = nullptr;
    char* src_data = nullptr;
    char* src_temp = nullptr;
    int data_offset = 0;
    int iter_cnt = src->bshape.n * src->bshape.c;

    int cidx = 0;
    int rank = 0;
    int bg = 0;
    int bank = 0;
    uint32_t col = 0;
    uint32_t row = 0;
    uint64_t addr = 0;
    uint32_t even_s_row = 0;  // starting_row;
    uint32_t even_s_col = 0;  // starting_col;
    uint32_t odd_s_row = 0;   // starting_row;
    uint32_t odd_s_col = 0;   // starting_col;

    int src_size = src->size;
    int type_size = (src->precision == PIM_FP16) ? 2 : 1;

    int out_cnt = 0;
    int in_cnt = 0;

    if (gemm_order_ == I_X_W) {
        out_cnt = src->bshape.w;
        in_cnt = src->bshape.h * type_size / trans_size;
    } else {
        out_cnt = src->bshape.h;
        in_cnt = src->bshape.w * type_size / trans_size;
    }

    if (src->bshape.w != src->bshape_r.w || src->bshape.h != src->bshape_r.h) {
        src_temp = (char*)calloc(src_size, sizeof(half_float::half));
        int copy_success;
        for (int i = 0; i < src->bshape_r.w; i++) {
            copy_success = copy_memory((half_float::half*)src_temp + i * src->bshape.h,
                                       (half_float::half*)src_data + i * src->bshape_r.h,
                                       src->bshape_r.h * sizeof(half_float::half), DEVICE_TO_HOST);
            if (copy_success != 0) {
                DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                return -1;
            }
        }
        copy_success = copy_memory(src_data, src_temp, src_size, HOST_TO_DEVICE);
        if (copy_success != 0) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
            return -1;
        }
        free(src_temp);
    }

    for (int iter = 0; iter < iter_cnt; iter++) {
        cidx = 0;
        rank = 0;
        bg = 0;
        bank = 0;
        col = 0;
        row = 0;
        addr = 0;
        even_s_row = 0;
        even_s_col = 0;
        odd_s_row = 0;
        odd_s_col = 0;
        dst_data = (char*)dst->data + data_offset;
        src_data = (char*)src->data + data_offset;

        for (int y = 0; y < out_cnt; y += out_tile_size) {
            for (int x = 0; x < in_cnt; x += in_tile_size) {
                if ((x / in_tile_size) % 2 == 0) {
                    for (int tiled_y = 0; tiled_y < out_tile_size; tiled_y += num_grf_B) {
                        col = even_s_col;
                        row = even_s_row;

                        for (int grfb_idx = 0; grfb_idx < num_grf_B; grfb_idx++) {
                            for (int grfa_idx = 0; grfa_idx < num_grf_A; grfa_idx++) {
                                addr = addr_gen_safe(cidx, rank, bg, bank, row, col);
#ifdef EMULATOR
                                int d_idx = (y + tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
#else
                                int d_idx = (y + tiled_y + grfb_idx) * in_cnt + x + grfa_idx;
#endif
                                memcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size);
                                col++;
                            }
                        }

                        bank += (num_banks / num_pim_blocks);

                        if (bank >= (num_banks / num_bank_groups)) {
                            bg++;
                            bank = 0;
                        }

                        if (bg >= num_bank_groups) {
                            bg = 0;
                            rank++;
                        }

                        if (rank >= num_pim_rank) {
                            rank = 0;
                            cidx++;
                        }

                        if (cidx >= num_pim_chan) {
                            cidx = 0;
                            even_s_row = row;
                            even_s_col = col;
                        }
                    }
                } else if ((x / in_tile_size) % 2 == 1) {
                    for (int tiled_y = 0; tiled_y < out_tile_size; tiled_y += num_grf_B) {
                        col = odd_s_col;
                        row = odd_s_row;

                        for (int grfb_idx = 0; grfb_idx < num_grf_B; grfb_idx++) {
                            for (int grfa_idx = 0; grfa_idx < num_grf_A; grfa_idx++) {
                                addr = addr_gen_safe(cidx, rank, bg, bank + 1, row, col);
#ifdef EMULATOR
                                int d_idx = (y + tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
#else
                                int d_idx = (y + tiled_y + grfb_idx) * in_cnt + x + grfa_idx;
#endif
                                memcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size);
                                col++;
                            }
                        }

                        bank += (num_banks / num_pim_blocks);

                        if (bank >= (num_banks / num_bank_groups)) {
                            bg++;
                            bank = 0;
                        }

                        if (bg >= num_bank_groups) {
                            bg = 0;
                            rank++;
                        }

                        if (rank >= num_pim_rank) {
                            rank = 0;
                            cidx++;
                        }

                        if (cidx >= num_pim_chan) {
                            cidx = 0;
                            odd_s_row = row;
                            odd_s_col = col;
                        }
                    }
                }
            }
        }
        data_offset += (src->bshape.h * src->bshape.w * sizeof(half_float::half));
    }
    dst->data_layout_type = PimDataLayoutType::ALIGNED_GEMM_WEIGHT;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

}  // namespace manager
}  // namespace runtime
}  // namespace pim
