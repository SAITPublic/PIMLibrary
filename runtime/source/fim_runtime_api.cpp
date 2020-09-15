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
bool fim_alloc_done = false;
uint64_t g_fim_base_addr = 0x0;
int FimInitialize(FimRuntimeType rt_type, FimPrecision precision)
{
    fim_alloc_done = false;
    if (!log_initialized) {
        google::InitGoogleLogging("FIMLibrary");
        FLAGS_minloglevel = FIM_LOG_LEVEL;
#if CONSOLE
        FLAGS_logtostderr = 1;
        FLAGS_stderrthreshold = 0;
#endif
        log_initialized = true;
    }
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(Initialize);
    int ret = 0;

    if (fim_runtime == nullptr) fim_runtime = new FimRuntime(rt_type, precision);
    ret = fim_runtime->initialize();
    FIM_PROFILE_TOCK(Initialize);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimDeinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(Deinitialize);
    int ret = 0;

    if (fim_runtime != nullptr) {
        ret = fim_runtime->deinitialize();
        delete fim_runtime;
        fim_runtime = nullptr;
    }
    FIM_PROFILE_TOCK(Deinitialize);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

FimBo* FimCreateBo(int w, int h, int c, int n, FimPrecision precision, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(CreateBo);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    FimBo* fim_bo = new FimBo;
    int type_size = (precision == FIM_FP16) ? 2 : 1;
    size_t size = w * h * c * n * type_size;

    fim_bo->size = size;
    fim_bo->bshape = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
    fim_bo->bshape_r = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
    fim_bo->mem_type = mem_type;
    fim_bo->precision = precision;

    ret = fim_runtime->alloc_memory(fim_bo);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    FIM_PROFILE_TOCK(CreateBo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return fim_bo;
}

FimBo* FimCreateBo(FimDesc* fim_desc, FimMemType mem_type, FimMemFlag mem_flag)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(CreateBo);

    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    FimBo* fim_bo = new FimBo;
    int type_size = (fim_desc->precision == FIM_FP16) ? 2 : 1;
    size_t size = get_aligned_size(fim_desc, mem_flag, fim_bo) * type_size;

    fim_bo->size = size;
    fim_bo->mem_type = mem_type;
    fim_bo->precision = fim_desc->precision;

    ret = fim_runtime->alloc_memory(fim_bo);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

#ifdef EMULATOR
    pad_data(fim_bo->data, fim_desc, mem_flag);
#endif

    FIM_PROFILE_TOCK(CreateBo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
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

    fim_desc->precision = precision;
    fim_desc->bshape_r = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
    align_shape(fim_desc, op_type);

    FIM_PROFILE_TOCK(CreateDesc);

    return fim_desc;
}

int FimDestroyBo(FimBo* fim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(DestroyBo);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->free_memory(fim_bo);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    delete fim_bo;
    FIM_PROFILE_TOCK(DestroyBo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimDestroyDesc(FimDesc* fim_desc)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(DestroyDesc);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        return -1;
    }
    delete fim_desc;
    FIM_PROFILE_TOCK(DestroyDesc);

    return ret;
}

int FimAllocMemory(void** ptr, size_t size, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(AllocMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->alloc_memory(ptr, size, mem_type);
    FIM_PROFILE_TOCK(AllocMemory);

    if (ptr == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimAllocMemory(FimBo* fim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(AllocMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->alloc_memory(fim_bo);
    FIM_PROFILE_TOCK(AllocMemory);

    if (fim_bo->data == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimFreeMemory(void* ptr, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
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
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(FreeMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->free_memory(fim_bo);
    FIM_PROFILE_TOCK(FreeMemory);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimConvertDataLayout(void* dst, void* src, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->convert_data_layout(dst, src, size, op_type);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimConvertDataLayout(FimBo* dst, FimBo* src, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->convert_data_layout(dst, src, op_type);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimConvertDataLayout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->convert_data_layout(dst, src0, src1, op_type);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimCopyMemory(void* dst, void* src, size_t size, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(CopyMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->copy_memory(dst, src, size, cpy_type);
    FIM_PROFILE_TOCK(CopyMemory);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimCopyMemory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(CopyMemory);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->copy_memory(dst, src, cpy_type);
    FIM_PROFILE_TOCK(CopyMemory);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecuteAdd(FimBo* output, FimBo* operand0, FimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteAdd);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute_add(output, operand0, operand1, stream, block);
    FIM_PROFILE_TOCK(ExecuteAdd);

    return ret;
}

int FimExecuteAdd(FimBo* output, void* scalar, FimBo* vector, void* stream, bool block)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteAdd);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }

    int num_vector = vector->size / sizeof(uint16_t);
    FimBo* padded_scalar = FimCreateBo(num_vector, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    for (int i = 0; i < num_vector; i++) {
        fim_runtime->copy_memory((void*)((char*)padded_scalar->data + i * sizeof(uint16_t)), scalar, sizeof(uint16_t),
                                 HOST_TO_FIM);
    }

    ret = fim_runtime->execute_add(output, vector, padded_scalar, stream, block);
    FIM_PROFILE_TOCK(ExecuteAdd);

    FimDestroyBo(padded_scalar);
    return ret;
}

int FimExecuteMul(FimBo* output, FimBo* operand0, FimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteMul);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }
    ret = fim_runtime->execute_mul(output, operand0, operand1, stream, block);
    FIM_PROFILE_TOCK(ExecuteMul);

    return ret;
}

int FimExecuteMul(FimBo* output, void* scalar, FimBo* vector, void* stream, bool block)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ExecuteMul);
    int ret = 0;

    if (fim_runtime == nullptr) {
        return -1;
    }

    int num_vector = vector->size / sizeof(uint16_t);
    FimBo* padded_scalar = FimCreateBo(num_vector, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    for (int i = 0; i < num_vector; i++) {
        fim_runtime->copy_memory((void*)((char*)padded_scalar->data + i * sizeof(uint16_t)), scalar, sizeof(uint16_t),
                                 HOST_TO_FIM);
    }

    ret = fim_runtime->execute_mul(output, vector, padded_scalar, stream, block);
    FIM_PROFILE_TOCK(ExecuteMul);

    FimDestroyBo(padded_scalar);
    return ret;
}

int FimExecuteGemv(FimBo* output, FimBo* operand0, FimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ExecuteGEMV);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->execute_gemv(output, operand0, operand1, stream, block);
    FIM_PROFILE_TOCK(ExecuteGEMV);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecuteGemvAdd(FimBo* output, FimBo* operand0, FimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ExecuteGEMVAdd);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    ret = fim_runtime->execute_gemv_add(output, operand0, operand1, stream, block);
    FIM_PROFILE_TOCK(ExecuteGEMVAdd);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecuteRelu(FimBo* output, FimBo* fim_data, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ExecuteRelu);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->execute_relu(output, fim_data, stream, block);
    FIM_PROFILE_TOCK(ExecuteRelu);

    return ret;
}

int FimExecuteBN(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance,
                 double epsilon, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ExecuteBN);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->execute_bn(output, fim_data, beta, gamma, mean, variance, epsilon, stream, block);
    FIM_PROFILE_TOCK(ExecuteBN);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimSynchronize(void* stream)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ExecuteSynchronize);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->execute_sync(stream);
    FIM_PROFILE_TOCK(ExecuteSynchronize);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecuteDummy(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(ExecuteDummy);
    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->execute_dummy();
    FIM_PROFILE_TOCK(ExecuteDummy);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

FimGemvBundle* FimCreateGemvBundle(FimBo* input, FimBo* weight, FimBo* output)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(CreateGemvBundle);

    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(ERROR) << "FimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    FimGemvBundle* bundle = new FimGemvBundle;

    bundle->in = input;
    bundle->wei = weight;
    bundle->out = output;

    FIM_PROFILE_TOCK(CreateGemvBundle);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return bundle;
}

FimGemvBundle* FimFindGemvBundle(uint64_t w_addr)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(FindGemvBundle);

    FimGemvBundle* addr = nullptr;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return addr;
    }
    addr = fim_runtime->find_gemv_bundle(w_addr);
    FIM_PROFILE_TOCK(FindGemvBundle);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return addr;
}

int FimInsertGemvBundle(uint64_t w_addr, FimGemvBundle* bundle)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FIM_PROFILE_TICK(InsertGemvBundle);

    int ret = 0;

    if (fim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = fim_runtime->insert_gemv_bundle(w_addr, bundle);
    FIM_PROFILE_TOCK(InsertGemvBundle);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}
