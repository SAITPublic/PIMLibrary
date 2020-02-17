#include "manager/FimManager.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimManager::FimManager(FimRuntimeType rt_type, FimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "called";
    fim_device_ = new FimDevice(precision_);
    fim_control_manager_ = new FimControlManager(fim_device_, rt_type_, precision_);
    fim_memory_manager_ = new FimMemoryManager(fim_device_, rt_type_, precision_);
}

FimManager::~FimManager(void)
{
    DLOG(INFO) << "called";
    delete fim_device_;
    delete fim_control_manager_;
    delete fim_memory_manager_;
}

FimManager* FimManager::get_instance(FimRuntimeType rt_type, FimPrecision precision)
{
    DLOG(INFO) << "called";
    static FimManager* instance_ = new FimManager(rt_type, precision);

    return instance_;
}

int FimManager::initialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    fim_device_->initialize();
    fim_control_manager_->initialize();
    fim_memory_manager_->initialize();

    return ret;
}

int FimManager::deinitialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    fim_memory_manager_->deinitialize();
    fim_control_manager_->deinitialize();
    fim_device_->deinitialize();

    return ret;
}

int FimManager::alloc_memory(void** ptr, size_t size, FimMemType mem_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->alloc_memory(ptr, size, mem_type);

    return ret;
}

int FimManager::alloc_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->alloc_memory(fim_bo);

    return ret;
}

int FimManager::free_memory(void* ptr, FimMemType mem_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->free_memory(ptr, mem_type);

    return ret;
}

int FimManager::free_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->free_memory(fim_bo);

    return ret;
}

int FimManager::copy_memory(void* dst, void* src, size_t size, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->copy_memory(dst, src, size, cpy_type);

    return ret;
}

int FimManager::copy_memory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->copy_memory(dst, src, cpy_type);

    return ret;
}

int FimManager::convert_data_layout(void* dst, void* src, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->convert_data_layout(dst, src, size, op_type);

    return ret;
}

int FimManager::convert_data_layout(FimBo* dst, FimBo* src, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->convert_data_layout(dst, src, op_type);

    return ret;
}

int FimManager::convert_data_layout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_memory_manager_->convert_data_layout(dst, src0, src1, op_type);

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
