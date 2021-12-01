/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "PimRuntime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "executor/PimExecutor.h"
#include "pim_runtime_api.h"
#include "utility/pim_log.h"
#include "utility/pim_util.h"

namespace pim
{
namespace runtime
{
PimRuntime::PimRuntime(PimRuntimeType rt_type, PimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    pim_manager_ = pim::runtime::manager::PimManager::get_instance(rt_type, precision);
    pim_executor_ = pim::runtime::executor::PimExecutor::get_instance(rt_type, precision);

    const char* env_k = std::getenv("PIM_KERNEL_TYPE");
    if (env_k != nullptr) {
        switch (*env_k) {
            case '1':
                kernel_type_ = PIM;
                break;
            case '2':
                kernel_type_ = CUSTOM_GPU;
                break;
            default:
                kernel_type_ = OPTIMAL;
        }
    }

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
        free_memory(it->second->wei);
        delete it->second;
    }

    pim_manager_->deinitialize();
    pim_executor_->deinitialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
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

int PimRuntime::convert_data_layout(void* dst, void* src, size_t size, PimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_manager_->convert_data_layout(dst, src, size, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::convert_data_layout(PimBo* dst, PimBo* src, PimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_manager_->convert_data_layout(dst, src, op_type);

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

    ret = pim_manager_->copy_memory(dst, src, cpy_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = pim_executor_->execute_add(output, operand0, operand1, (hipStream_t)stream, block);

    return ret;
}

int PimRuntime::execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = pim_executor_->execute_mul(output, operand0, operand1, (hipStream_t)stream, block);

    return ret;
}

int PimRuntime::execute_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (kernel_type_ == CUSTOM_GPU) {
        ret = pim_executor_->execute_custom_gemv(output, operand0, operand1, false, (hipStream_t)stream, block);
    } else if (kernel_type_ == PIM && !is_transposed(operand1)) {
        PimGemvBundle* bundle = get_gemv_bundle(operand1, operand0, output);
        operand1 = bundle->wei;
        ret = pim_executor_->execute_gemv(output, operand0, operand1, (hipStream_t)stream, block);
    } else {
        if (is_pim_available(output, operand0, operand1, OP_GEMV)) {
            PimGemvBundle* bundle = get_gemv_bundle(operand1, operand0, output);
            operand1 = bundle->wei;
            ret = pim_executor_->execute_gemv(output, operand0, operand1, (hipStream_t)stream, block);
        } else {
            ret = pim_executor_->execute_custom_gemv(output, operand0, operand1, false, (hipStream_t)stream, block);
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (kernel_type_ == CUSTOM_GPU) {
        ret = pim_executor_->execute_custom_gemv(output, operand0, operand1, true, (hipStream_t)stream, block);
    } else if (kernel_type_ == PIM && !is_transposed(operand1)) {
        PimGemvBundle* bundle = get_gemv_bundle(operand1, operand0, output);
        operand1 = bundle->wei;
        ret = pim_executor_->execute_gemv_add(output, operand0, operand1, (hipStream_t)stream, block);
    } else {
        if (is_pim_available(output, operand0, operand1, OP_GEMV)) {
            PimGemvBundle* bundle = get_gemv_bundle(operand1, operand0, output);
            operand1 = bundle->wei;
            ret = pim_executor_->execute_gemv_add(output, operand0, operand1, (hipStream_t)stream, block);
        } else {
            ret = pim_executor_->execute_custom_gemv(output, operand0, operand1, true, (hipStream_t)stream, block);
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, PimBo* operand2, bool relu,
                                 void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret =
        pim_executor_->execute_custom_gemv_add(output, operand0, operand1, operand2, relu, (hipStream_t)stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_relu(output, pim_data, (hipStream_t)stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                           double epsilon, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_bn(output, pim_data, beta, gamma, mean, variance, epsilon, (hipStream_t)stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_sync(void* stream)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_sync((hipStream_t)stream);

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

PimGemvBundle* PimRuntime::find_gemv_bundle(PimBo* weight)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PimGemvBundle* addr = nullptr;

#if 0
    char env_p, *ptr;
    bool need_flush = false;
    if (std::getenv("ENABLE_MIOPEN_PYTORCH")) {
      ptr = std::getenv("ENABLE_MIOPEN_PYTORCH");
      env_p = *ptr;
    } else {
      env_p = '0';
    }
    if (env_p == '1')
      need_flush = true;

    if (need_flush && (weight->mem_type == MEM_TYPE_DEVICE || weight->mem_type == MEM_TYPE_PIM)) {
        /* GPU cache should be flushed before CPU access to the area */
        int cache_flush;
        hipMemcpy(&cache_flush, weight->data, sizeof(int), hipMemcpyDeviceToHost);
    }
#endif

    uint32_t w_key = 0;
    uint32_t* w_addr_ptr = reinterpret_cast<uint32_t*>(weight->data);
    int step = weight->size >> 1;

    for (int i = 0; i < weight->size / sizeof(uint32_t); i += step) {
        w_key ^= w_addr_ptr[i];
    }
    std::unordered_map<uint32_t, PimGemvBundle*>::const_iterator found = weight_map_.find(w_key);
    if (found != weight_map_.end()) {
        addr = found->second;
    } else {
        DLOG(INFO) << "[%s] not found\tw_addr:%p, w_key:%X, weight_map_size:%d\n" << __func__ << weight->data << w_key << weight_map_.size();
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return addr;
}

int PimRuntime::insert_gemv_bundle(PimBo* weight, PimGemvBundle* bundle)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    uint32_t w_key = 0;
    uint32_t* w_addr_ptr = reinterpret_cast<uint32_t*>(weight->data);
    int step = weight->size >> 1;
    for (int i = 0; i < weight->size / sizeof(uint32_t); i += step) {
        w_key ^= w_addr_ptr[i];
    }
    weight_map_.insert(std::make_pair(w_key, bundle));
    printf("[%s] insert\tw_addr:%p, w_key:%X, weight_map_size:%d\n", __func__, weight->data, w_key,
           weight_map_.size());

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

PimGemvBundle* PimRuntime::get_gemv_bundle(PimBo* weight, PimBo* dev_in, PimBo* dev_out)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    // TODO: change the key from uint64_t to (keybo*, size): pull_req: 456)?
    uint64_t w_addr = reinterpret_cast<uint64_t>(weight->data);
    PimGemvBundle* bundle = nullptr;
    bundle = find_gemv_bundle(weight);

    if (bundle == nullptr) {
        PimDesc* pim_desc =
            PimCreateDesc(weight->bshape_r.n, 1, weight->bshape_r.h, weight->bshape_r.w, PIM_FP16, OP_GEMV);
        PimBo* host_weight = nullptr;

        if (weight->data == nullptr) {
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return nullptr;
        }
        if (weight->mem_type == MEM_TYPE_HOST) {
            host_weight = weight;
        } else if (weight->mem_type == MEM_TYPE_DEVICE || weight->mem_type == MEM_TYPE_PIM) {
            uint32_t w_size = weight->bshape_r.h * weight->bshape_r.w;
            host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
            hipMemcpy(host_weight->data, weight->data, w_size * sizeof(uint16_t), hipMemcpyDeviceToHost);
        }

        PimBo* host_reordered_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
        PimBo* pre_wei = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);

        convert_data_layout(host_reordered_weight, host_weight, OP_GEMV);
        PimCopyMemory(pre_wei, host_reordered_weight, HOST_TO_PIM);

        bundle = new PimGemvBundle;
        bundle->in = dev_in;
        bundle->wei = pre_wei;
        bundle->out = dev_out;

        insert_gemv_bundle(weight, bundle);

        PimDestroyDesc(pim_desc);
        PimDestroyBo(host_reordered_weight);

        if (host_weight != weight) {
            PimDestroyBo(host_weight);
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return bundle;
}

} /* namespace runtime */
} /* namespace pim */
