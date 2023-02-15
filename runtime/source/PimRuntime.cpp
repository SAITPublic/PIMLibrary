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

#include "PimRuntime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "executor/IPimExecutor.h"
#include "executor/PimCompilerDriver.h"
#include "executor/PimExecutorFactory.h"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_util.h"

using namespace pim::runtime::pimc_driver;

namespace pim
{
namespace runtime
{
PimRuntime::PimRuntime(PimRuntimeType rt_type, PimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    pim_manager_ = manager::PimManager::get_instance(rt_type, precision);
    pim_executor_ = executor::PimExecutorFactory::getPimExecutor(pim_manager_, this, rt_type, precision);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

PimRuntime::~PimRuntime(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    pim_executor_.reset();
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

int PimRuntime::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    pim_manager_->initialize();
    pim_executor_->initialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    for (auto it = weight_map_.begin(); it != weight_map_.end(); ++it) {
        free_memory(it->second);
        delete it->second;
    }

    pim_manager_->deinitialize();
    pim_executor_->deinitialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::set_device(uint32_t device_id)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->set_device(device_id);
    if (ret != 0) {
        DLOG(ERROR) << "Failed to set device " << ret << "Device ID: " << device_id << std::endl;
        return ret;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::get_device(uint32_t* device_id)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->get_device(device_id);
    if (ret != 0) {
        DLOG(ERROR) << "Failed to get device " << ret << "Device ID: " << device_id << std::endl;
        return ret;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

void* PimRuntime::create_stream(PimRuntimeType rt_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    void* stream_obj = NULL;
    if (rt_type == RT_TYPE_HIP) {
        stream_obj = pim_executor_->create_stream();
    } else {
        DLOG(ERROR) << __FUNCTION__ << " not implemented for runtime " << rt_type << " \n";
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return stream_obj;
}

int PimRuntime::alloc_memory(void** ptr, size_t size, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_manager_->alloc_memory(ptr, size, mem_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::alloc_memory(PimBo* pim_bo, void* user_ptr)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (!user_ptr) {
        ret = pim_manager_->alloc_memory(pim_bo);
        if (ret != 0) {
            DLOG(ERROR) << "Fail to alloc memory";
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            ret = -1;
        }
        pim_bo->use_user_ptr = false;
    } else {
        pim_bo->data = user_ptr;
        pim_bo->use_user_ptr = true;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::free_memory(void* ptr, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_manager_->free_memory(ptr, mem_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::free_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (!pim_bo->use_user_ptr) {
        ret = pim_manager_->free_memory(pim_bo);
        if (ret != 0) {
            DLOG(ERROR) << "Fail to free PIM buffer object";
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_manager_->copy_memory(dst, src, size, cpy_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (cpy_type == PIM_TO_PIM) {
        ret = pim_executor_->execute_copy(dst, src, NULL, true);
    } else {
        ret = pim_manager_->copy_memory(dst, src, cpy_type);
    }
    dst->data_layout_type = src->data_layout_type;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::copy_memory_3d(const PimCopy3D* copy_params)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_manager_->copy_memory_3d(copy_params);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = pim_executor_->execute_add(output, operand0, operand1, stream, block);

    return ret;
}

int PimRuntime::execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = pim_executor_->execute_mul(output, operand0, operand1, stream, block);

    return ret;
}

int PimRuntime::execute_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                             PimGemmOrder gemm_order, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    pim_executor_->set_gemm_order(gemm_order);
    ret = pim_executor_->execute_gemm(output, input, weight, bias, act_func, stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_relu(output, pim_data, stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                           double epsilon, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_bn(output, pim_data, beta, gamma, mean, variance, epsilon, stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_sync(void* stream)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_sync(stream);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_dummy(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_dummy();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

PimBo* PimRuntime::find_preloaded_pim_weight(PimBo* weight)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PimBo* addr = nullptr;

    uint32_t w_key = 0;
    uint32_t* w_addr_ptr = reinterpret_cast<uint32_t*>(weight->data);
    int step = weight->size >> 1;

    for (int i = 0; i < weight->size / sizeof(uint32_t); i += step) {
        w_key ^= w_addr_ptr[i];
    }
    std::unordered_map<uint32_t, PimBo*>::const_iterator found = weight_map_.find(w_key);
    if (found != weight_map_.end()) {
        addr = found->second;
    } else {
        DLOG(INFO) << "[%s] not found\tw_addr:%p, w_key:%X, weight_map_size:%d\n"
                   << __func__ << weight->data << w_key << weight_map_.size();
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return addr;
}

int PimRuntime::insert_preloaded_pim_weight(PimBo* dev_wei, PimBo* pim_wei)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    uint32_t w_key = 0;
    uint32_t* w_addr_ptr = reinterpret_cast<uint32_t*>(dev_wei->data);
    int step = dev_wei->size >> 1;
    for (int i = 0; i < dev_wei->size / sizeof(uint32_t); i += step) {
        w_key ^= w_addr_ptr[i];
    }
    weight_map_.insert(std::make_pair(w_key, pim_wei));
    DLOG(INFO) << "[%s] insert\tw_addr:%p, w_key:%X, weight_map_size:%lu\n"
               << __func__ << dev_wei->data << w_key << weight_map_.size();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

bool PimRuntime::check_need_for_transpose(PimGemmOrder gemm_order, PimBo* dev_wei)
{
    bool transposed = dev_wei->transposed;
    bool ret = false;

    if (gemm_order == I_X_W) {
        if (transposed == false)
            return true;
        else
            return false;
    } else {
        if (transposed == false)
            return false;
        else
            return true;
    }

    return ret;
}

PimBo* PimRuntime::get_preloaded_pim_gemm_weight(PimBo* dev_wei, PimGemmOrder gemm_order, bool reorder_on_device,
                                                 void* stream, bool save_for_reuse)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PimBo* pre_wei = find_preloaded_pim_weight(dev_wei);

    if (pre_wei == nullptr) {
        if (reorder_on_device) {
            pim_manager_->set_gemm_order(gemm_order);
            pre_wei = PimCreateBo(dev_wei->bshape.n, dev_wei->bshape.c, dev_wei->bshape.h, dev_wei->bshape.w, PIM_FP16,
                                  MEM_TYPE_PIM);
            pim_manager_->convert_data_layout(pre_wei, dev_wei, true, stream);
        } else {
            PimBo* host_weight = nullptr;
            PimBo* host_weight_t = nullptr;
            PimBo* host_reordered_weight = nullptr;
            PimBShape* bshape = &dev_wei->bshape;

            if (dev_wei->data == nullptr) {
                DLOG(ERROR) << "[END] " << __FUNCTION__ << " called";
                return nullptr;
            }

            host_weight = PimCreateBo(bshape->n, bshape->c, bshape->h, bshape->w, PIM_FP16, MEM_TYPE_HOST);
            host_weight_t = PimCreateBo(bshape->n, bshape->c, bshape->h, bshape->w, PIM_FP16, MEM_TYPE_HOST);
            host_reordered_weight = PimCreateBo(bshape->n, bshape->c, bshape->h, bshape->w, PIM_FP16, MEM_TYPE_HOST);
            pre_wei = PimCreateBo(bshape->n, bshape->c, bshape->h, bshape->w, PIM_FP16, MEM_TYPE_PIM);

            if (check_need_for_transpose(gemm_order, dev_wei) == true) {
                pim_manager_->copy_memory(host_weight_t, dev_wei, DEVICE_TO_HOST);
                transpose_pimbo(host_weight, host_weight_t);
            } else {
                pim_manager_->copy_memory(host_weight, dev_wei, DEVICE_TO_HOST);
            }

            pim_manager_->set_gemm_order(gemm_order);
            pim_manager_->convert_data_layout(host_reordered_weight, host_weight, false, nullptr);
            PimCopyMemory(pre_wei, host_reordered_weight, HOST_TO_PIM);

            PimDestroyBo(host_weight);
            PimDestroyBo(host_weight_t);
            PimDestroyBo(host_reordered_weight);
        }

        if (save_for_reuse) {
            insert_preloaded_pim_weight(dev_wei, pre_wei);
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pre_wei;
}

PimBo* PimRuntime::generate_gemm_weight_from_buffer(PimBo* src, PimGemmOrder gemm_order, bool reorder_on_device,
                                                    void* stream, bool save_for_reuse)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    PimBo* pre_weight = nullptr;
    if (src->data_layout_type == PimDataLayoutType::RAW && is_pim_applicable(src, gemm_order)) {
        pre_weight = get_preloaded_pim_gemm_weight(src, gemm_order, reorder_on_device, stream, save_for_reuse);
    } else {
        DLOG(ERROR) << "GEMM weight generation for provided layout is not supported yet";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    PimBo* pim_reordered_buff;
    // Make sure reordered data are stored on the PIM memory
    if (pre_weight->mem_type == MEM_TYPE_PIM) {
        pim_reordered_buff = pre_weight;
    } else {
        const auto direction = pre_weight->mem_type == MEM_TYPE_HOST ? HOST_TO_PIM : DEVICE_TO_PIM;
        pim_reordered_buff = PimCreateBo(pre_weight->bshape.n, pre_weight->bshape.c, pre_weight->bshape.h,
                                         pre_weight->bshape.w, pre_weight->precision, MEM_TYPE_PIM, nullptr);
        PimCopyMemory(pim_reordered_buff, pre_weight, direction);
        if (pre_weight != src) {
            PimDestroyBo(pre_weight);
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_reordered_buff;
}

#if PIM_COMPILER_ENABLE == 1
PimCompiledObj* PimRuntime::build_program(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs,
                                          std::vector<PimBo*> input_pimbo, PimTarget* target, std::string compile_opts)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    auto pimc_driver = Driver::create_driver(target);
    PimCompiledObj* pim_co = pimc_driver->build_program(output, inputs, input_pimbo, target, compile_opts);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_co;
}

PimBo* PimRuntime::execute_program(PimCompiledObj* obj, PimTarget* target, std::string launch_opts)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    auto pimc_driver = Driver::create_driver(target);
    PimBo* output_pimbo = pimc_driver->execute_program(obj, target, launch_opts);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return output_pimbo;
}
#endif

} /* namespace runtime */
} /* namespace pim */
