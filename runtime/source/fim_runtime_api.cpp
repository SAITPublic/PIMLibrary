#include "fim_runtime_api.h"
#include <iostream>
#include "FimRuntime.h"
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"
#include "utility/fim_profile.h"

using namespace fim::runtime;

FimRuntime* fim_runtime = nullptr;
static bool log_initialized = false;

int FimInitialize(FimRuntimeType rt_type, FimPrecision precision)
{
    if (!log_initialized) {
        FLAGS_logtostderr = true;
        FLAGS_minloglevel = FIM_LOG_LEVEL;
        google::InitGoogleLogging("FIMRuntime");
        log_initialized = true;
    }

    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Initialize);
    int ret = 0;

    if (fim_runtime == nullptr) fim_runtime = new FimRuntime(rt_type, precision);
    ret = fim_runtime->initialize();
    FIM_PROFILE_TOCK(Initialize);

    return ret;
}

int FimDeinitialize(void)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Deinitialize);
    int ret = 0;

    if (fim_runtime != nullptr) {
        ret = fim_runtime->deinitialize();
        delete fim_runtime;
        fim_runtime = nullptr;
    }
    FIM_PROFILE_TOCK(Deinitialize);

    return ret;
}

FimBo* FimCreateBo(int w, int h, int c, int n, FimPrecision precision, FimMemType mem_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(CreateBo);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        return nullptr;
    }

    FimBo* fim_bo = new FimBo;
    int type_size = (precision == FIM_FP16) ? 2 : 1;
    size_t size = w * h * c * n * type_size;

    fim_bo->size = size;
    fim_bo->bshape = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
    fim_bo->mem_type = mem_type;
    fim_bo->precision = precision;

    ret = fim_runtime->alloc_memory(fim_bo);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        return nullptr;
    }
    FIM_PROFILE_TOCK(CreateBo);

    return fim_bo;
}

int FimDestroyBo(FimBo* fim_bo)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(DestroyBo);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        return -1;
    }
    ret = fim_runtime->free_memory(fim_bo);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        return -1;
    }
    delete fim_bo;
    FIM_PROFILE_TOCK(DestroyBo);

    return ret;
}

int FimAllocMemory(void** ptr, size_t size, FimMemType mem_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(AllocMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->alloc_memory(ptr, size, mem_type);
    FIM_PROFILE_TOCK(AllocMemory);

    if (ptr == nullptr) return -1;

    return ret;
}

int FimAllocMemory(FimBo* fim_bo)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(AllocMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->alloc_memory(fim_bo);
    FIM_PROFILE_TOCK(AllocMemory);

    if (fim_bo->data == nullptr) return -1;

    return ret;
}

int FimFreeMemory(void* ptr, FimMemType mem_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(FreeMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->free_memory(ptr, mem_type);
    FIM_PROFILE_TOCK(FreeMemory);

    return ret;
}

int FimFreeMemory(FimBo* fim_bo)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(FreeMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->free_memory(fim_bo);
    FIM_PROFILE_TOCK(FreeMemory);

    return ret;
}

int FimConvertDataLayout(void* dst, void* src, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->convert_data_layout(dst, src, size, op_type);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    return ret;
}

int FimConvertDataLayout(FimBo* dst, FimBo* src, FimOpType op_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->convert_data_layout(dst, src, op_type);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    return ret;
}

int FimConvertDataLayout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->convert_data_layout(dst, src0, src1, op_type);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    return ret;
}

int FimCopyMemory(void* dst, void* src, size_t size, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(CopyMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->copy_memory(dst, src, size, cpy_type);
    FIM_PROFILE_TOCK(CopyMemory);

    return ret;
}

int FimCopyMemory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(CopyMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->copy_memory(dst, src, cpy_type);
    FIM_PROFILE_TOCK(CopyMemory);

    return ret;
}

int FimExecute(void* output, void* operand0, void* operand1, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Execute);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute(output, operand0, operand1, size, op_type);
    FIM_PROFILE_TOCK(Execute);

    return ret;
}

int FimExecute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType op_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Execute);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute(output, operand0, operand1, op_type);
    FIM_PROFILE_TOCK(Execute);

    return ret;
}

int FimExecute(FimBo* output, FimBo* fim_data, FimOpType op_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Execute);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute(output, fim_data, op_type);
    FIM_PROFILE_TOCK(Execute);

    return ret;
}

int FimExecuteBN(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* scale, FimBo* shift)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteBN);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute_bn(output, fim_data, beta, gamma, scale, shift);
    FIM_PROFILE_TOCK(ExecuteBN);

    return ret;
}
