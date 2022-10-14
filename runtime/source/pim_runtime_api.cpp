/*
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "pim_runtime_api.h"
#include <iostream>
#include "PimRuntime.h"
#include "executor/PimCompilerDriver.h"
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

using namespace pim::runtime;

std::unique_ptr<PimRuntime> pim_runtime;
static bool log_initialized = false;
static bool pim_initialized = false;
bool pim_alloc_done[10] = {false};
uint64_t g_pim_base_addr[10] = {0x0};

int PimInitialize(PimRuntimeType rt_type, PimPrecision precision)
{
    int ret = 0;

    if (!log_initialized) {
#if CONSOLE
        FLAGS_logtostderr = 1;
        FLAGS_stderrthreshold = 0;
#endif
        google::InitGoogleLogging("PIMLibrary");
        FLAGS_minloglevel = PIM_LOG_LEVEL;
        log_initialized = true;
    }

    if (!pim_initialized) {
        DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
        PIM_PROFILE_TICK(Initialize);
        pim_runtime = std::make_unique<PimRuntime>(rt_type, precision);
        ret = pim_runtime->initialize();
        pim_initialized = true;
        PIM_PROFILE_TOCK(Initialize);
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    } else {
        DLOG(INFO) << "PIM Already initialized " << __FUNCTION__ << " called";
    }

    return ret;
}

int PimDeinitialize(void)
{
    int ret = 0;

    if (pim_initialized) {
        DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
        PIM_PROFILE_TICK(Deinitialize);
        ret = pim_runtime->deinitialize();
        pim_runtime.reset();
        pim_initialized = false;
        PIM_PROFILE_TOCK(Deinitialize);
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    } else {
        DLOG(INFO) << "PIM has been already deinitialized " << __FUNCTION__ << " called";
    }

    return ret;
}

int PimSetDevice(uint32_t device_id)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(PimSetDevice);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->set_device(device_id);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to Set Device";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return ret;
    }

    PIM_PROFILE_TOCK(PimSetDevice);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimGetDevice(uint32_t* device_id)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(PimGetDevice);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->get_device(device_id);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to Set Device";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return ret;
    }

    PIM_PROFILE_TOCK(PimGetDevice);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

PimBo* PimCreateBo(int n, int c, int h, int w, PimPrecision precision, PimMemType mem_type, void* user_ptr,
                   bool transposed)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(CreateBo);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    PimBo* pim_bo = new PimBo;
    int type_size = (precision == PIM_FP16) ? 2 : 1;
    size_t size = n * c * h * w * type_size;

    pim_bo->size = size;
    pim_bo->bshape = {(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w};
    pim_bo->bshape_r = {(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w};
    pim_bo->mem_type = mem_type;
    pim_bo->precision = precision;
    pim_bo->transposed = transposed;
    pim_bo->data_layout_type = PimDataLayoutType::RAW;

    ret = pim_runtime->alloc_memory(pim_bo, user_ptr);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    PIM_PROFILE_TOCK(CreateBo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_bo;
}

PimBo* PimCreateBo(PimDesc* pim_desc, PimMemType mem_type, PimMemFlag mem_flag, void* user_ptr)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(CreateBo);

    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    PimBo* pim_bo = new PimBo;
    int type_size = (pim_desc->precision == PIM_FP16) ? 2 : 1;
    size_t size = get_aligned_size(pim_desc, mem_flag, pim_bo) * type_size;

    pim_bo->size = size;
    pim_bo->mem_type = mem_type;
    pim_bo->precision = pim_desc->precision;
    pim_bo->data_layout_type = PimDataLayoutType::RAW;

    ret = pim_runtime->alloc_memory(pim_bo, user_ptr);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        LOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    PIM_PROFILE_TOCK(CreateBo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_bo;
}

PimBo* PimCreateBo(PimGemmDesc* pim_gemm_desc, PimMemType mem_type, PimMemFlag mem_flag, void* user_ptr)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(CreateBo);

    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    PimBo* pim_bo = new PimBo;

    set_pimbo(pim_gemm_desc, mem_type, mem_flag, pim_bo);

    ret = pim_runtime->alloc_memory(pim_bo, user_ptr);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to alloc memory";
        LOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    PIM_PROFILE_TOCK(CreateBo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_bo;
}

PimGemmDesc* PimCreateGemmDesc(int n, int c, int inout_h, int in_w, int out_w, PimPrecision precision, bool transposed)
{
    DLOG(INFO) << "called";

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        return nullptr;
    }

    PimGemmDesc* pim_gemm_desc = new PimGemmDesc;

    pim_gemm_desc->precision = precision;
    pim_gemm_desc->transposed = transposed;
    pim_gemm_desc->in_bshape_r = {(uint32_t)n, (uint32_t)c, (uint32_t)inout_h, (uint32_t)in_w};
    pim_gemm_desc->wei_bshape_r = {(uint32_t)n, (uint32_t)c, (uint32_t)in_w, (uint32_t)out_w};
    pim_gemm_desc->bias_bshape_r = {(uint32_t)n, (uint32_t)c, (uint32_t)inout_h, (uint32_t)out_w};
    pim_gemm_desc->out_bshape_r = {(uint32_t)n, (uint32_t)c, (uint32_t)inout_h, (uint32_t)out_w};

    align_gemm_shape(pim_gemm_desc);

    return pim_gemm_desc;
}

int PimDestroyGemmDesc(PimGemmDesc* pim_gemm_desc)
{
    DLOG(INFO) << "called";

    int ret = 0;

    if (pim_gemm_desc == nullptr) {
        DLOG(ERROR) << "pim_gemm_desc is nullptr";
        return -1;
    }

    delete pim_gemm_desc;

    return ret;
}

PimDesc* PimCreateDesc(int n, int c, int h, int w, PimPrecision precision, PimOpType op_type)
{
    DLOG(INFO) << "called";
    PIM_PROFILE_TICK(CreateDesc);

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        return nullptr;
    }

    PimDesc* pim_desc = new PimDesc;

    pim_desc->precision = precision;
    pim_desc->bshape_r = {(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w};
    pim_desc->op_type = op_type;

    align_shape(pim_desc, op_type);

    PIM_PROFILE_TOCK(CreateDesc);

    return pim_desc;
}

int PimDestroyBo(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(DestroyBo);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    ret = pim_runtime->free_memory(pim_bo);
    if (ret != 0) {
        DLOG(ERROR) << "Fail to Destroy Bo";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    delete pim_bo;
    PIM_PROFILE_TOCK(DestroyBo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimDestroyDesc(PimDesc* pim_desc)
{
    DLOG(INFO) << "called";
    PIM_PROFILE_TICK(DestroyDesc);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        return -1;
    }
    delete pim_desc;
    PIM_PROFILE_TOCK(DestroyDesc);

    return ret;
}

int PimAllocMemory(void** ptr, size_t size, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(AllocMemory);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->alloc_memory(ptr, size, mem_type);
    PIM_PROFILE_TOCK(AllocMemory);

    if (ptr == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimAllocMemory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(AllocMemory);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->alloc_memory(pim_bo);
    PIM_PROFILE_TOCK(AllocMemory);

    if (pim_bo->data == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimFreeMemory(void* ptr, PimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(FreeMemory);
    int ret = 0;

    if (pim_runtime == nullptr) {
        return -1;
    }
    ret = pim_runtime->free_memory(ptr, mem_type);
    PIM_PROFILE_TOCK(FreeMemory);

    return ret;
}

int PimFreeMemory(PimBo* pim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(FreeMemory);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->free_memory(pim_bo);
    PIM_PROFILE_TOCK(FreeMemory);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimCopyMemory(void* dst, void* src, size_t size, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(CopyMemory);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->copy_memory(dst, src, size, cpy_type);
    PIM_PROFILE_TOCK(CopyMemory);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimCopyMemory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(CopyMemory);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->copy_memory(dst, src, cpy_type);
    PIM_PROFILE_TOCK(CopyMemory);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimCopyMemoryRect(const PimCopy3D* copy_params)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(CopyMemoryRect);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->copy_memory_3d(copy_params);
    PIM_PROFILE_TOCK(CopyMemoryRect);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

void* createStream(PimRuntimeType rt_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return NULL;
    }
    void* new_stream = NULL;
    new_stream = pim_runtime->createStream(rt_type);
    if (new_stream == NULL) {
        DLOG(ERROR) << "unable to create stream for runtime " << rt_type << "\n";
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return new_stream;
}

int PimExecuteAdd(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    PIM_PROFILE_TICK(ExecuteAdd);
    int ret = 0;

    if (pim_runtime == nullptr) {
        return -1;
    }
    ret = pim_runtime->execute_add(output, operand0, operand1, stream, block);
    PIM_PROFILE_TOCK(ExecuteAdd);

    return ret;
}

int PimExecuteAdd(PimBo* output, void* scalar, PimBo* vector, void* stream, bool block)
{
    DLOG(INFO) << "called";
    PIM_PROFILE_TICK(ExecuteAdd);
    int ret = 0;

    if (pim_runtime == nullptr) {
        return -1;
    }

    int num_vector = vector->size / sizeof(uint16_t);
    PimBo* padded_scalar = PimCreateBo(1, 1, 1, num_vector, PIM_FP16, MEM_TYPE_PIM);
    for (int i = 0; i < num_vector; i++) {
        pim_runtime->copy_memory((void*)((char*)padded_scalar->data + i * sizeof(uint16_t)), scalar, sizeof(uint16_t),
                                 HOST_TO_PIM);
    }

    ret = pim_runtime->execute_add(output, vector, padded_scalar, stream, block);
    PIM_PROFILE_TOCK(ExecuteAdd);

    PimDestroyBo(padded_scalar);
    std::cout << "PimDestryBo called in " << __FUNCTION__ << std::endl;
    return ret;
}

int PimExecuteMul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    PIM_PROFILE_TICK(ExecuteMul);
    int ret = 0;

    if (pim_runtime == nullptr) {
        return -1;
    }
    ret = pim_runtime->execute_mul(output, operand0, operand1, stream, block);
    PIM_PROFILE_TOCK(ExecuteMul);

    return ret;
}

int PimExecuteMul(PimBo* output, void* scalar, PimBo* vector, void* stream, bool block)
{
    DLOG(INFO) << "called";
    PIM_PROFILE_TICK(ExecuteMul);
    int ret = 0;

    if (pim_runtime == nullptr) {
        return -1;
    }

    int num_vector = vector->size / sizeof(uint16_t);
    PimBo* padded_scalar = PimCreateBo(1, 1, 1, num_vector, PIM_FP16, MEM_TYPE_PIM);
    for (int i = 0; i < num_vector; i++) {
        pim_runtime->copy_memory((void*)((char*)padded_scalar->data + i * sizeof(uint16_t)), scalar, sizeof(uint16_t),
                                 HOST_TO_PIM);
    }

    ret = pim_runtime->execute_mul(output, vector, padded_scalar, stream, block);
    PIM_PROFILE_TOCK(ExecuteMul);

    PimDestroyBo(padded_scalar);
    std::cout << "PimDestryBo called in " << __FUNCTION__ << std::endl;
    return ret;
}

int PimExecuteGemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                   bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(ExecuteGemm);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    ret = pim_runtime->execute_gemm(output, input, weight, bias, act_func, stream, block);
    PIM_PROFILE_TOCK(ExecuteGemm);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecuteRelu(PimBo* output, PimBo* pim_data, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(ExecuteRelu);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->execute_relu(output, pim_data, stream, block);
    PIM_PROFILE_TOCK(ExecuteRelu);

    return ret;
}

int PimExecuteBN(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                 double epsilon, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(ExecuteBN);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->execute_bn(output, pim_data, beta, gamma, mean, variance, epsilon, stream, block);
    PIM_PROFILE_TOCK(ExecuteBN);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimSynchronize(void* stream)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(ExecuteSynchronize);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->execute_sync(stream);
    PIM_PROFILE_TOCK(ExecuteSynchronize);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecuteDummy(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(ExecuteDummy);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    ret = pim_runtime->execute_dummy();
    PIM_PROFILE_TOCK(ExecuteDummy);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

PimBo* PimConvertGemmWeight(PimBo* src, bool save_for_reuse)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(PimGetReorderedBuffer);

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    PimBo* dst = pim_runtime->generate_gemm_weight_from_buffer(src, save_for_reuse);
    if (dst == nullptr) {
        DLOG(ERROR) << "Failed to reorder source buffer";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    PIM_PROFILE_TOCK(PimGetReorderedBuffer);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return dst;
}

#if PIM_COMPILER_ENABLE == 1
PimTarget* PimCreateTarget(PimRuntimeType rt_type = PimRuntimeType::RT_TYPE_HIP,
                           PimPrecision precision = PimPrecision::PIM_FP16, PimDevice device = PimDevice::GPU)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(CreateTarget);

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    PimTarget* target = new PimTarget;
    target->runtime = rt_type;
    target->precision = precision;
    target->device = device;

    PIM_PROFILE_TOCK(CreateTarget);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return target;
}

int PimDestroyTarget(PimTarget* target)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(DestroyTarget);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        return -1;
    }
    delete target;

    PIM_PROFILE_TOCK(DestroyTarget);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

PimCompiledObj* PimBuildProgram(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs,
                                std::vector<PimBo*> input_pimbo, PimTarget* target, std::string compile_opts)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(PimBuildProgram);

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    PimCompiledObj* pim_co = pim_runtime->build_program(output, inputs, input_pimbo, target, compile_opts);
    if (pim_co == nullptr) {
        DLOG(ERROR) << "Failed to build program, compile";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    PIM_PROFILE_TOCK(PimBuildProgram);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_co;
}

PimBo* PimExecuteProgram(PimCompiledObj* obj, PimTarget* target, std::string launch_opts)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(PimExecuteProgram);

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    PimBo* output_pimbo = pim_runtime->execute_program(obj, target, launch_opts);
    if (output_pimbo == nullptr) {
        DLOG(ERROR) << "Failed to execute program";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    PIM_PROFILE_TOCK(PimExecuteProgram);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return output_pimbo;
}

int PimDestroyProgram(PimCompiledObj* obj)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(DestroyProgram);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(ERROR) << "PimRuntime is not initialized";
        return -1;
    }
    obj->input_pimbo.clear();
    obj->input_pimbo.shrink_to_fit();
    obj->new_pimbo.clear();
    obj->new_pimbo.shrink_to_fit();
    obj->op_order.clear();
    obj->op_order.shrink_to_fit();
    obj->pimbo_map.clear();
    delete obj;

    PIM_PROFILE_TOCK(DestroyProgram);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}
#endif
