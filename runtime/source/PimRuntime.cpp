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
#include "utility/pim_log.h"

namespace pim
{
namespace runtime
{
PimRuntime::PimRuntime(PimRuntimeType rt_type, PimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    pim_manager_ = pim::runtime::manager::PimManager::get_instance(rt_type, precision);
    pim_executor_ = pim::runtime::executor::PimExecutor::get_instance(rt_type, precision);

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
        free_memory(it->second->in);
        free_memory(it->second->wei);
        free_memory(it->second->out);
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

int PimRuntime::alloc_memory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_manager_->alloc_memory(pim_bo);

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

    ret = pim_manager_->free_memory(pim_bo);

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

    ret = pim_executor_->execute_gemv(output, operand0, operand1, (hipStream_t)stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_gemv_add(output, operand0, operand1, (hipStream_t)stream, block);

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

PimGemvBundle* PimRuntime::find_gemv_bundle(uint64_t w_addr)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PimGemvBundle* addr = nullptr;

    std::unordered_map<uint64_t, PimGemvBundle*>::const_iterator found = weight_map_.find(w_addr);
    if (found != weight_map_.end()) {
        addr = found->second;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return addr;
}

int PimRuntime::insert_gemv_bundle(uint64_t w_addr, PimGemvBundle* bundle)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    std::unordered_map<uint64_t, PimGemvBundle*>::const_iterator found = weight_map_.find(w_addr);
    if (found == weight_map_.end()) {
        weight_map_.insert(std::make_pair(w_addr, bundle));
    } else {
        ret = -1;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

} /* namespace runtime */
} /* namespace pim */
