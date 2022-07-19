/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/HIPMemManager.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "manager/PimMemoryManager.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

#ifdef __cplusplus
extern "C" {
#endif
uint64_t fmm_map_pim(uint32_t, uint32_t, uint64_t);
#ifdef __cplusplus
}
#endif

namespace pim
{
namespace runtime
{
namespace manager
{
HIPMemManager::HIPMemManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision)
    : PimMemoryManager(pim_device, rt_type, precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

HIPMemManager::~HIPMemManager() { DLOG(INFO) << "[START] " << __FUNCTION__ << " called"; }
int HIPMemManager::initialize()
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int device_id_ = PimMemoryManager::initialize();
    if (device_id_ < 0) {
        return device_id_;
    }
    hipGetDeviceCount(&num_gpu_devices_);
    if (device_id_ != num_gpu_devices_) {
        ret = -1;
        DLOG(ERROR) << "Number of GPU Ids and Device Count doesn't match" << __FUNCTION__ << " called";
    }

    for (int device = 0; device < num_gpu_devices_; device++) {
        fragment_allocator_.push_back(new SimpleHeap<PimBlockAllocator>);
    }
    hipGetDevice(&device_id_);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HIPMemManager::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = PimMemoryManager::deinitialize();
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HIPMemManager::alloc_memory(void** ptr, size_t size, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)ptr, size) != hipSuccess) {
            return -1;
        }
    } else if (mem_type == MEM_TYPE_HOST) {
        if (hipHostMalloc((void**)ptr, size) != hipSuccess) {
            return -1;
        }
    } else if (mem_type == MEM_TYPE_PIM) {
        hipGetDevice(&device_id_);
        *ptr = fragment_allocator_[device_id_]->alloc(size, device_id_, rt_type_);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HIPMemManager::alloc_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (pim_bo->mem_type == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)&pim_bo->data, pim_bo->size) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (pim_bo->mem_type == MEM_TYPE_HOST) {
        if (hipHostMalloc((void**)&pim_bo->data, pim_bo->size) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (pim_bo->mem_type == MEM_TYPE_PIM) {
        hipGetDevice(&device_id_);
        pim_bo->data = fragment_allocator_[device_id_]->alloc(pim_bo->size, device_id_, rt_type_);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HIPMemManager::free_memory(void* ptr, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        if (hipFree(ptr) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (mem_type == MEM_TYPE_HOST) {
        hipHostFree(ptr);
    } else if (mem_type == MEM_TYPE_PIM) {
        hipGetDevice(&device_id_);
        return fragment_allocator_[device_id_]->free(ptr);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HIPMemManager::free_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (pim_bo->mem_type == MEM_TYPE_DEVICE) {
        if (hipFree(pim_bo->data) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    } else if (pim_bo->mem_type == MEM_TYPE_HOST) {
        hipHostFree(pim_bo->data);
        pim_bo->data = nullptr;
    } else if (pim_bo->mem_type == MEM_TYPE_PIM) {
        hipGetDevice(&device_id_);
        if (fragment_allocator_[device_id_]->free(pim_bo->data)) return 0;
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HIPMemManager::copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type)
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
    } else if (cpy_type == PIM_TO_PIM || cpy_type == DEVICE_TO_PIM || cpy_type == PIM_TO_DEVICE ||
               cpy_type == DEVICE_TO_DEVICE) {
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

int HIPMemManager::copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type)
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

size_t PrecisionSize(const PimBo* bo)
{
    assert(bo != nullptr && "Invalid buffer");
    switch (bo->precision) {
        case PIM_FP16:
            return sizeof(half_float::half);
        case PIM_INT8:
        default:
            return 1ul;
    }
}

int HIPMemManager::copy_memory_3d(const PimCopy3D* copy_params)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    hipMemcpy3DParms param;
    param.srcArray = nullptr;
    param.dstArray = nullptr;

    param.srcPos = make_hipPos(copy_params->src_x_in_bytes, copy_params->src_y, copy_params->src_z);  // x, y, z

    param.dstPos = make_hipPos(copy_params->dst_x_in_bytes, copy_params->dst_y, copy_params->dst_z);

    param.extent = make_hipExtent(copy_params->width_in_bytes, copy_params->height, copy_params->depth);  // w, h, d

    // error check remaining - xsz = pitch/sizeof(precision)
    if (copy_params->src_mem_type == MEM_TYPE_HOST &&
        (copy_params->dst_mem_type == MEM_TYPE_DEVICE || copy_params->dst_mem_type == MEM_TYPE_PIM)) {
        param.kind = hipMemcpyHostToDevice;
        param.srcPtr = make_hipPitchedPtr((void*)copy_params->src_ptr, copy_params->src_pitch, copy_params->src_pitch,
                                          copy_params->src_height);  // d, pitch, xsz, ysz

        auto* bo = copy_params->dst_bo;
        param.dstPtr = make_hipPitchedPtr((void*)bo->data, bo->bshape.w * PrecisionSize(bo),
                                          bo->bshape.w * PrecisionSize(bo), bo->bshape.h);

    } else if ((copy_params->src_mem_type == MEM_TYPE_DEVICE || copy_params->src_mem_type == MEM_TYPE_PIM) &&
               copy_params->dst_mem_type == MEM_TYPE_HOST) {
        param.kind = hipMemcpyDeviceToHost;
        auto* bo = copy_params->src_bo;
        param.srcPtr = make_hipPitchedPtr((void*)bo->data, bo->bshape.w * PrecisionSize(bo),
                                          bo->bshape.w * PrecisionSize(bo), bo->bshape.h);
        param.dstPtr = make_hipPitchedPtr((void*)copy_params->dst_ptr, copy_params->dst_pitch, copy_params->dst_pitch,
                                          copy_params->dst_height);

    } else if ((copy_params->src_mem_type == MEM_TYPE_DEVICE || copy_params->src_mem_type == MEM_TYPE_PIM) &&
               (copy_params->dst_mem_type == MEM_TYPE_DEVICE || copy_params->dst_mem_type == MEM_TYPE_PIM)) {
        param.kind = hipMemcpyDeviceToDevice;

        auto* sbo = copy_params->src_bo;
        param.srcPtr = make_hipPitchedPtr((void*)sbo->data, sbo->bshape.w * PrecisionSize(sbo),
                                          sbo->bshape.w * PrecisionSize(sbo), sbo->bshape.h);

        auto* dbo = copy_params->dst_bo;
        param.dstPtr = make_hipPitchedPtr((void*)dbo->data, dbo->bshape.w * PrecisionSize(dbo),
                                          dbo->bshape.w * PrecisionSize(dbo), dbo->bshape.h);
    } else if (copy_params->src_mem_type == MEM_TYPE_HOST && copy_params->dst_mem_type == MEM_TYPE_HOST) {
        param.kind = hipMemcpyHostToHost;
        param.srcPtr = make_hipPitchedPtr((void*)copy_params->src_ptr, copy_params->src_pitch, copy_params->src_pitch,
                                          copy_params->src_height);
        param.dstPtr = make_hipPitchedPtr((void*)copy_params->dst_ptr, copy_params->dst_pitch, copy_params->dst_pitch,
                                          copy_params->dst_height);
    }

    if (hipMemcpy3D(&param) != hipSuccess) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    return ret;
}

int HIPMemManager::convert_data_layout(void* dst, void* src, size_t size, PimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    DLOG(ERROR) << "not yet implemented";

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HIPMemManager::convert_data_layout(PimBo* dst, PimBo* src, PimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    bool is_chwise = false;
    int ch_per_op = 0;

    if (src->bshape.n % fbi_.num_pim_chan == 0) {
        is_chwise = true;
        ch_per_op = 1;
    }

    if (op_type == OP_GEMV) {
        if (is_chwise == true)
            ret = convert_data_layout_for_gemv_weight(dst, src, ch_per_op);
        else
            ret = convert_data_layout_for_gemv_weight(dst, src);
    } else {
        DLOG(ERROR) << "not yet implemented";
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HIPMemManager::convert_data_layout_for_gemv_weight(PimBo* dst, PimBo* src)
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
            if (hipMemcpy((half*)src_temp + i * src->bshape.w, (half*)src_data + i * src->bshape_r.w,
                          src->bshape_r.w * sizeof(half), hipMemcpyDeviceToHost) != hipSuccess) {
                DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                return -1;
            }
        }
        if (hipMemcpy(src_data, src_temp, src->size, hipMemcpyHostToDevice) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
            return -1;
        }
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
                            if (hipMemcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size,
                                          hipMemcpyDeviceToDevice) != hipSuccess) {
                                DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                                return -1;
                            }
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
                            if (hipMemcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size,
                                          hipMemcpyDeviceToDevice) != hipSuccess) {
                                DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                                return -1;
                            }
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

int HIPMemManager::convert_data_layout_for_gemv_weight(PimBo* dst, PimBo* src, int ch_per_op)
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
    int batch = src->bshape.n;
    int origin_width = src->bshape_r.w;
    int padded_width = src->bshape.w;
    int origin_height = src->bshape_r.h;
    int padded_height = src->bshape.h;
    int weight_batch = src->bshape_r.n;

    int origin_matrix_dim = origin_width * origin_height;
    int padded_matrix_dim = padded_width * padded_height;
    int transaction_width_dim = padded_width * type_size / trans_size;
    int tile_width_dim = num_grf_A;
    int tile_height_dim = num_grf_B * num_pim_blocks * ch_per_op;  // ex) 64* 2;

    int padded_idx;
    int origin_idx;
    int start_channel = 0;
    int end_channel = ch_per_op;

    half* dst_half = (half*)dst->data;
    half* src_half = (half*)src->data;

    if (padded_width != origin_width || padded_height != origin_height) {
        src_temp = (char*)calloc(src->size / sizeof(half), sizeof(half));

        for (int b = 0; b < weight_batch; b++) {
            for (int y = 0; y < padded_height; y++) {
                padded_idx = b * padded_matrix_dim + y * padded_width;
                origin_idx = b * origin_matrix_dim + y * origin_width;
                if (hipMemcpy((half*)src_temp + padded_idx, (half*)src_data + origin_idx, origin_width * sizeof(half),
                              hipMemcpyDeviceToHost) != hipSuccess) {
                    DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                    return -1;
                }
            }
        }

        if (hipMemcpy(src_data, src_temp, src->size, hipMemcpyHostToDevice) != hipSuccess) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
            return -1;
        }

        free(src_temp);
    }

    for (int b = 0; b < batch; b++) {
        for (int y = 0; y < padded_height; y += tile_height_dim) {
            for (int x = 0; x < transaction_width_dim; x += tile_width_dim) {
                if ((x / tile_width_dim) % 2 == 0) {
                    for (int tiled_y = 0; tiled_y < tile_height_dim; tiled_y += num_grf_B) {
                        col = even_s_col;
                        row = even_s_row;
                        for (int grfb_idx = 0; grfb_idx < num_grf_B; grfb_idx++) {
                            for (int grfa_idx = 0; grfa_idx < num_grf_A; grfa_idx++) {
                                addr = addr_gen_safe(start_channel + cidx, rank, bg, bank, row, col);
#ifdef EMULATOR
                                int d_idx = b * padded_height * transaction_width_dim +
                                            (y + tiled_y + grfa_idx) * transaction_width_dim + x + grfb_idx;
#else
                                int d_idx = b * padded_height * transaction_width_dim +
                                            (y + tiled_y + grfb_idx) * transaction_width_dim + x + grfa_idx;
#endif
                                if (hipMemcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size,
                                              hipMemcpyDeviceToDevice) != hipSuccess) {
                                    DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                                    return -1;
                                }
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

                        if (start_channel + cidx == end_channel) {
                            cidx = 0;
                            even_s_row = row;
                            even_s_col = col;
                        }
                    }
                } else if ((x / tile_width_dim) % 2 == 1) {
                    for (int tiled_y = 0; tiled_y < tile_height_dim; tiled_y += num_grf_B) {
                        col = odd_s_col;
                        row = odd_s_row;

                        for (int grfb_idx = 0; grfb_idx < num_grf_B; grfb_idx++) {
                            for (int grfa_idx = 0; grfa_idx < num_grf_A; grfa_idx++) {
                                addr = addr_gen_safe(start_channel + cidx, rank, bg, bank + 1, row, col);
#ifdef EMULATOR
                                int d_idx = b * padded_height * transaction_width_dim +
                                            (y + tiled_y + grfa_idx) * transaction_width_dim + x + grfb_idx;
#else
                                int d_idx = b * padded_height * transaction_width_dim +
                                            (y + tiled_y + grfb_idx) * transaction_width_dim + x + grfa_idx;
#endif

                                if (hipMemcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size,
                                              hipMemcpyDeviceToDevice) != hipSuccess) {
                                    DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                                    return -1;
                                }
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

                        if (start_channel + cidx == end_channel) {
                            cidx = 0;
                            odd_s_row = row;
                            odd_s_col = col;
                        }
                    }
                }
            }
        }

        start_channel = start_channel + ch_per_op;
        end_channel = end_channel + ch_per_op;
        cidx = 0;
        even_s_row = 0;
        even_s_col = 0;
        odd_s_row = 0;
        odd_s_col = 0;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

}  // namespace manager
}  // namespace runtime
}  // namespace pim
