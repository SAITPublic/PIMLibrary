#include "manager/FimManager.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "executor/fim_hip_kernels/fim_crf_bins.h"
#include "utility/fim_log.h"
#include "utility/fim_util.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimManager::FimManager(FimRuntimeType rt_type, FimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    fim_device_ = new FimDevice(precision_);
    fim_control_manager_ = new FimControlManager(fim_device_, rt_type_, precision_);
    fim_memory_manager_ = new FimMemoryManager(fim_device_, rt_type_, precision_);
    fim_crf_generator_ = new FimCrfBinGen();
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

FimManager::~FimManager(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    delete fim_device_;
    delete fim_control_manager_;
    delete fim_memory_manager_;
    delete fim_crf_generator_;
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

FimManager* FimManager::get_instance(FimRuntimeType rt_type, FimPrecision precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    static FimManager* instance_ = new FimManager(rt_type, precision);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return instance_;
}

int FimManager::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    fim_device_->initialize();
    fim_control_manager_->initialize();
    fim_memory_manager_->initialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    fim_memory_manager_->deinitialize();
    fim_control_manager_->deinitialize();
    fim_device_->deinitialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::alloc_memory(void** ptr, size_t size, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->alloc_memory(ptr, size, mem_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::alloc_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->alloc_memory(fim_bo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::free_memory(void* ptr, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->free_memory(ptr, mem_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::free_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->free_memory(fim_bo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::copy_memory(void* dst, void* src, size_t size, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->copy_memory(dst, src, size, cpy_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::copy_memory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->copy_memory(dst, src, cpy_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::convert_data_layout(void* dst, void* src, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->convert_data_layout(dst, src, size, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::convert_data_layout(FimBo* dst, FimBo* src, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->convert_data_layout(dst, src, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::convert_data_layout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_memory_manager_->convert_data_layout(dst, src0, src1, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimManager::create_crf_binary(FimOpType op_type, int input_size, int output_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    get_fim_block_info(&fbi_);
    fim_crf_generator_->gen_binary(op_type, input_size, output_size, &fbi_, h_binary_buffer_, &crf_size_);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

uint8_t* FimManager::get_crf_binary() { return h_binary_buffer_; }
int FimManager::get_crf_size() { return crf_size_; }
} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
