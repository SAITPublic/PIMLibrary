/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/hip/HipMemoryManager.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <list>
#include "hip/hip_runtime.h"
#include "manager/HostInfo.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_util.h"

extern std::map<uint32_t, HostInfo*> host_devices;

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

HipMemoryManager::HipMemoryManager(std::shared_ptr<PimDevice> pim_device, PimPrecision precision)
    : pim_device_(pim_device), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    pbi_ = pim_device_->get_pim_block_info();

    int max_topology = 32;
    FILE* fd;
    char path[256];
    uint32_t gpu_id;
    int host_cnt = 0;
    std::list<int> hip_visible_devices = get_env("HIP_VISIBLE_DEVICES");
    hipGetDeviceCount(&num_gpu_devices_);

    // if hip_device is not set , then assume all devices are visible
    if (hip_visible_devices.empty()) {
        for (int device = 0; device < num_gpu_devices_; device++) {
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
            host_devices[host_cnt] = host_info;
            host_cnt++;
            hip_visible_devices.pop_front();
        }
        curr++;
    }

    if (host_cnt == 0) {
        DLOG(ERROR) << "AMDGPU device not found " << __FUNCTION__ << " called";
        return;
    }

    if (host_cnt != num_gpu_devices_) {
        DLOG(ERROR) << "Number of GPU Ids and Device Count doesn't match" << __FUNCTION__ << " called";
        return;
    }

    for (int device = 0; device < num_gpu_devices_; device++) {
        fragment_allocator_.push_back(std::make_shared<SimpleHeap<HipBlockAllocator>>());
    }
    hipGetDevice(&host_id_);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

HipMemoryManager::~HipMemoryManager(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    pim_device_.reset();
    fragment_allocator_.clear();
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

int HipMemoryManager::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HipMemoryManager::deinitialize(void)
{
    int ret = 0;
    return ret;
}

int HipMemoryManager::alloc_memory(void** ptr, size_t size, PimMemType mem_type)
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
        hipGetDevice(&host_id_);
        *ptr = fragment_allocator_[host_id_]->alloc(size, host_id_);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HipMemoryManager::alloc_memory(PimBo* pim_bo)
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
        hipGetDevice(&host_id_);
        pim_bo->data = fragment_allocator_[host_id_]->alloc(pim_bo->size, host_id_);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HipMemoryManager::free_memory(void* ptr, PimMemType mem_type)
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
        hipGetDevice(&host_id_);
        return fragment_allocator_[host_id_]->free(ptr);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HipMemoryManager::free_memory(PimBo* pim_bo)
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
        hipGetDevice(&host_id_);
        if (fragment_allocator_[host_id_]->free(pim_bo->data)) return 0;
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HipMemoryManager::copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type)
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

int HipMemoryManager::copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type)
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

int HipMemoryManager::copy_memory_3d(const PimCopy3D* copy_params)
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

int HipMemoryManager::convert_data_layout(PimBo* dst, PimBo* src)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    bool is_chwise = check_chwise_gemm_bo(src);

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

int HipMemoryManager::convert_data_layout_for_chwise_gemm_weight(PimBo* dst, PimBo* src)
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
    int in_cnt = src->bshape.h * type_size / trans_size;

    int in_tile_size = num_grf_A;
    int out_tile_size = num_grf_B * num_pim_blocks * num_pim_chan * num_pim_rank;
    int data_offset = 0;
    int iter_cnt = src->bshape.n * src->bshape.c * src->bshape.w / PIM_GEMV_OUT_ALIGN;

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
                            int d_idx = (tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
#else
                            int d_idx = (tiled_y + grfb_idx) * in_cnt + x + grfa_idx;
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
        data_offset += (src->bshape.h * PIM_GEMV_OUT_ALIGN * sizeof(half));
    }
    dst->data_layout_type = PimDataLayoutType::CHWISE_GEMM_WEIGHT;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int HipMemoryManager::convert_data_layout_for_aligned_gemm_weight(PimBo* dst, PimBo* src)
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

    int type_size = (src->precision == PIM_FP16) ? 2 : 1;
    int out_cnt = src->bshape.w;
    int in_cnt = src->bshape.h * type_size / trans_size;
    int src_size = src->size;

    if (src->bshape.w != src->bshape_r.w || src->bshape.h != src->bshape_r.h) {
        src_temp = (char*)calloc(src_size, sizeof(half));
        for (int i = 0; i < src->bshape_r.w; i++) {
            if (hipMemcpy((half*)src_temp + i * src->bshape.h, (half*)src->data + i * src->bshape_r.h,
                          src->bshape_r.h * sizeof(half), hipMemcpyDeviceToHost) != hipSuccess) {
                DLOG(INFO) << "[END] " << __FUNCTION__ << " Failed to copy";
                return -1;
            }
        }
        if (hipMemcpy(src->data, src_temp, src_size, hipMemcpyHostToDevice) != hipSuccess) {
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
        data_offset += (src->bshape.h * src->bshape.w * sizeof(half));
    }
    dst->data_layout_type = PimDataLayoutType::ALIGNED_GEMM_WEIGHT;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

}  // namespace manager
}  // namespace runtime
}  // namespace pim
