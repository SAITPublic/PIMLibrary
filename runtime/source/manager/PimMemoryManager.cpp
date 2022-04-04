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

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "hip/hip_runtime.h"
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

PimMemoryManager::~PimMemoryManager(void) { DLOG(INFO) << "[START] " << __FUNCTION__ << " called"; }
int PimMemoryManager::initialize(void)
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

    hipGetDeviceCount(&num_gpu_devices_);

    if (device_id != num_gpu_devices_) {
        ret = -1;
        DLOG(ERROR) << "Number of GPU Ids and Device Count doesn't match" << __FUNCTION__ << " called";
    }

    for (int device = 0; device < num_gpu_devices_; device++) {
        fragment_allocator_.push_back(new SimpleHeap<PimBlockAllocator>);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::alloc_memory(void** ptr, size_t size, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)ptr, size) != hipSuccess) {
            return -1;
        }
    } else if (mem_type == MEM_TYPE_HOST) {
#ifdef ROCM3
        *ptr = (void*)malloc(size);
#else
        if (hipHostMalloc((void**)ptr, size) != hipSuccess) {
            return -1;
        }
#endif
    } else if (mem_type == MEM_TYPE_PIM) {
        int device_id = 0;
        hipGetDevice(&device_id);
        *ptr = fragment_allocator_[device_id]->alloc(size);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::alloc_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (pim_bo->mem_type == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)&pim_bo->data, pim_bo->size) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (pim_bo->mem_type == MEM_TYPE_HOST) {
#ifdef ROCM3
        pim_bo->data = (void*)malloc(pim_bo->size);
#else
        if (hipHostMalloc((void**)&pim_bo->data, pim_bo->size) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
#endif
    } else if (pim_bo->mem_type == MEM_TYPE_PIM) {
        int device_id = 0;
        hipGetDevice(&device_id);
        pim_bo->data = fragment_allocator_[device_id]->alloc(pim_bo->size);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::free_memory(void* ptr, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        if (hipFree(ptr) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (mem_type == MEM_TYPE_HOST) {
#ifdef ROCM3
        if (nullptr != ptr) free(ptr);
        ptr = nullptr;
#else
        hipHostFree(ptr);
#endif
    } else if (mem_type == MEM_TYPE_PIM) {
        int device_id = 0;
        hipGetDevice(&device_id);
        return fragment_allocator_[device_id]->free(ptr);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::free_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (pim_bo->mem_type == MEM_TYPE_DEVICE) {
        if (hipFree(pim_bo->data) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (pim_bo->mem_type == MEM_TYPE_HOST) {
#ifdef ROCM3
        if (nullptr != pim_bo->data) free(pim_bo->data);
        pim_bo->data = nullptr;
#else
        hipHostFree(pim_bo->data);
        pim_bo->data = nullptr;
#endif
    } else if (pim_bo->mem_type == MEM_TYPE_PIM) {
        int device_id = 0;
        hipGetDevice(&device_id);
        if (fragment_allocator_[device_id]->free(pim_bo->data)) return 0;
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (cpy_type == HOST_TO_PIM || cpy_type == HOST_TO_DEVICE) {
        if (hipMemcpy(dst, src, size, hipMemcpyHostToDevice) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (cpy_type == PIM_TO_HOST || cpy_type == DEVICE_TO_HOST) {
        if (hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (cpy_type == DEVICE_TO_PIM || cpy_type == PIM_TO_DEVICE || cpy_type == DEVICE_TO_DEVICE) {
        if (hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (cpy_type == HOST_TO_HOST) {
        if (hipMemcpy(dst, src, size, hipMemcpyHostToHost) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    }

    return ret;
}

int PimMemoryManager::copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    size_t size = dst->size;

    if (cpy_type == HOST_TO_PIM || cpy_type == HOST_TO_DEVICE) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyHostToDevice) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (cpy_type == PIM_TO_HOST || cpy_type == DEVICE_TO_HOST) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyDeviceToHost) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (cpy_type == DEVICE_TO_PIM || cpy_type == PIM_TO_DEVICE || cpy_type == DEVICE_TO_DEVICE) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyDeviceToDevice) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (cpy_type == HOST_TO_HOST) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyHostToHost) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::convert_data_layout(void* dst, void* src, size_t size, PimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    DLOG(ERROR) << "not yet implemented";

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::convert_data_layout(PimBo* dst, PimBo* src, PimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (op_type == OP_GEMV) {
        ret = convert_data_layout_for_gemv_weight(dst, src);
    } else {
        DLOG(ERROR) << "not yet implemented";
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimMemoryManager::convert_data_layout_for_gemv_weight(PimBo* dst, PimBo* src)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    int num_grf_A = fbi_.num_grf;
    int num_grf_B = fbi_.num_grf;
    int num_pim_blocks = fbi_.num_pim_blocks;
    int num_pim_chan = fbi_.num_pim_chan;
    int num_pim_rank = fbi_.num_pim_rank;
    int num_banks = fbi_.num_banks;
    int num_bank_groups = fbi_.num_bank_groups;
    int trans_size = fbi_.trans_size;

    int in_tile_size = num_grf_A;
    int out_tile_size = num_grf_B * num_pim_blocks * num_pim_chan * num_pim_rank;
    char* dst_data = (char*)dst->data;
    char* src_data = (char*)src->data;
    char* src_temp;

    int cidx = 0;
    int rank = 0;
    int bg = 0;
    int bank = 0;
    uint32_t col = 0;
    uint32_t row = 0;
    uint64_t addr;
    uint32_t even_s_row = 0;  // starting_row;
    uint32_t even_s_col = 0;  // starting_col;
    uint32_t odd_s_row = 0;   // starting_row;
    uint32_t odd_s_col = 0;   // starting_col;

    int type_size = (src->precision == PIM_FP16) ? 2 : 1;
    int out_cnt = src->bshape.h;
    int in_cnt = src->bshape.w * type_size / trans_size;

    if (src->bshape.w != src->bshape_r.w || src->bshape.h != src->bshape_r.h) {
        src_temp = (char*)calloc(src->size / sizeof(half), sizeof(half));
        for (int i = 0; i < src->bshape_r.h; i++) {
#ifdef ROCM3
            memcpy((half*)src_temp + i * src->bshape.w, (half*)src_data + i * src->bshape_r.w,
                   src->bshape_r.w * sizeof(half));
#else
            if (hipMemcpy((half*)src_temp + i * src->bshape.w, (half*)src_data + i * src->bshape_r.w,
                          src->bshape_r.w * sizeof(half), hipMemcpyDeviceToHost) != hipSuccess) {
                DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                return -1;
            }
#endif
        }
#if ROCM3
        memcpy(src_data, src_temp, src->size);
#else
        if (hipMemcpy(src_data, src_temp, src->size, hipMemcpyHostToDevice) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
            return -1;
        }
#endif
        free(src_temp);
    }

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
#if ROCM3
                            memcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size);
#else
                            if (hipMemcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size,
                                          hipMemcpyDeviceToDevice) != hipSuccess) {
                                DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                                return -1;
                            }
#endif
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
#if ROCM3
                            memcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size);
#else
                            if (hipMemcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size,
                                          hipMemcpyDeviceToDevice) != hipSuccess) {
                                DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                                return -1;
                            }
#endif
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

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

void* PimBlockAllocator::alloc(size_t request_size, size_t& allocated_size) const
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    assert(request_size <= block_size() && "BlockAllocator alloc request exceeds block size.");
    uint64_t ret = 0;
    size_t bsize = block_size();

    ret = allocate_pim_block(bsize);

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

uint64_t PimBlockAllocator::allocate_pim_block(size_t bsize) const
{
    uint64_t ret = 0;
    int device_id = 0;
    hipGetDevice(&device_id);
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
#ifndef ROCM3
#ifndef EMULATOR
        hipHostRegister((void*)g_pim_base_addr[device_id], bsize, hipRegisterExternalSvm);
#endif
#endif
    } else {
        std::cout << "fmm_map_pim failed! " << ret << std::endl;
    }
    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */
