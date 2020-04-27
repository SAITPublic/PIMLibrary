#include "fim_runtime_api.h"
#include <iostream>
#include "FimRuntime.h"
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"
#include "utility/fim_profile.h"
#include "utility/fim_util.h"

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

FimBo* FimCreateBo(FimDesc* fim_desc, FimMemType mem_type, FimMemFlag mem_flag)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(CreateBo);

    int data_type = fim_desc->precision;
    int ret = 0;

    FimBo* fim_bo = new FimBo;
    int type_size = (data_type == FIM_FP16) ? 2 : 1;
    size_t size = GetPaddedSize(fim_desc, mem_flag) * type_size;

    fim_bo->size = size;
    fim_bo->mem_type = mem_type;

    ret = fim_runtime->alloc_memory(fim_bo);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        return nullptr;
    }

    if (mem_flag == GEMV_INPUT) {
        PadInputData(fim_bo->data, fim_desc->bshape_r.w, fim_desc->bshape.w, mem_flag);
    }

    return fim_bo;
}

FimDesc* FimCreateDesc(int n, int c, int h, int w, FimPrecision precision, FimOpType op_type)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(CreateDesc);

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        return nullptr;
    }

    FimDesc* fim_desc = new FimDesc;

    fim_desc->op_type = op_type;
    fim_desc->bshape_r = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
    fim_desc->bshape = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
    fim_desc->precision = precision;

    FIM_PROFILE_TOCK(CreateDesc);

    return fim_desc;
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

int FimDestroyDesc(FimDesc* fim_desc)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(DestroyBo);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        return -1;
    }
    delete fim_desc;
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

int FimConvertDataLayout(FimBo* dst, FimBo* src, FimOpType op_type, FimDesc* fim_desc)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->convert_data_layout(dst, src, op_type, fim_desc);
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

int FimExecuteAdd(FimBo* output, FimBo* fim_data)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteAdd);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute_add(output, fim_data);
    FIM_PROFILE_TOCK(ExecuteAdd);

    return ret;
}

int FimExecuteMul(FimBo* output, FimBo* fim_data)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteMul);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute_mul(output, fim_data);
    FIM_PROFILE_TOCK(ExecuteMul);

    return ret;
}

int FimExecuteGEMV(FimBo* output, FimBo* operand0, FimBo* operand1, FimDesc* fim_desc)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteGEMV);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute_gemv(output, operand0, operand1, fim_desc);
    FIM_PROFILE_TOCK(ExecuteGEMV);

    return ret;
}

int FimExecuteRelu(FimBo* output, FimBo* fim_data)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteRelu);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute_relu(output, fim_data);
    FIM_PROFILE_TOCK(ExecuteRelu);

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
