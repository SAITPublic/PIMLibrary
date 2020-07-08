#include "FimRuntime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "executor/FimExecutor.h"
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
FimRuntime::FimRuntime(FimRuntimeType rt_type, FimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    fim_manager_ = fim::runtime::manager::FimManager::get_instance(rt_type, precision);
    fim_executor_ = fim::runtime::executor::FimExecutor::get_instance(rt_type, precision);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

int FimRuntime::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    fim_manager_->initialize();
    fim_executor_->initialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    fim_manager_->deinitialize();
    fim_executor_->deinitialize();

    for (auto it = weight_map_.begin(); it != weight_map_.end(); ++it) {
        free_memory(it->second->in, MEM_TYPE_DEVICE);
        free_memory(it->second->wei, MEM_TYPE_FIM);
        delete it->second->in;
        delete it->second->wei;
        delete it->second;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::alloc_memory(void** ptr, size_t size, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->alloc_memory(ptr, size, mem_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::alloc_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->alloc_memory(fim_bo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::free_memory(void* ptr, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->free_memory(ptr, mem_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::free_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->free_memory(fim_bo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::convert_data_layout(void* dst, void* src, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->convert_data_layout(dst, src, size, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::convert_data_layout(FimBo* dst, FimBo* src, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->convert_data_layout(dst, src, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::convert_data_layout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->convert_data_layout(dst, src0, src1, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::copy_memory(void* dst, void* src, size_t size, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->copy_memory(dst, src, size, cpy_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::copy_memory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->copy_memory(dst, src, cpy_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_add(FimBo* output, FimBo* operand0, FimBo* operand1, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_executor_->execute_add(output, operand0, operand1, block);

    return ret;
}

int FimRuntime::execute_mul(FimBo* output, FimBo* operand0, FimBo* operand1, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_executor_->execute_mul(output, operand0, operand1, block);

    return ret;
}

int FimRuntime::execute_gemv(FimBo* output, FimBo* operand0, FimBo* operand1, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_gemv(output, operand0, operand1, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_gemv_add(FimBo* output, FimBo* operand0, FimBo* operand1, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_gemv_add(output, operand0, operand1, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_relu(FimBo* output, FimBo* fim_data, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_relu(output, fim_data, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_bn(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance,
                           double epsilon, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_bn(output, fim_data, beta, gamma, mean, variance, epsilon, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_sync()
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_sync();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_dummy(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_dummy();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

FimGemvBundle* FimRuntime::find_gemv_bundle(uint64_t w_addr)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FimGemvBundle* addr = nullptr;

    std::unordered_map<uint64_t, FimGemvBundle*>::const_iterator found = weight_map_.find(w_addr);
    if (found != weight_map_.end()) {
        addr = found->second;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return addr;
}

int FimRuntime::insert_gemv_bundle(uint64_t w_addr, FimGemvBundle* fim_addr)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    std::unordered_map<uint64_t, FimGemvBundle*>::const_iterator found = weight_map_.find(w_addr);
    if (found == weight_map_.end()) {
        weight_map_.insert(std::make_pair(w_addr, fim_addr));
    } else {
        ret = -1;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

} /* namespace runtime */
} /* namespace fim */
