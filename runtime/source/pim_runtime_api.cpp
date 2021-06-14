/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
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
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

using namespace pim::runtime;

PimRuntime* pim_runtime = nullptr;
static bool log_initialized = false;
static bool pim_initialized = false;
bool pim_alloc_done = false;
uint64_t g_pim_base_addr = 0x0;
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

        if (pim_runtime == nullptr) pim_runtime = new PimRuntime(rt_type, precision);
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
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(Deinitialize);
    int ret = 0;

    if (pim_runtime != nullptr) {
        ret = pim_runtime->deinitialize();
        delete pim_runtime;
        pim_runtime = nullptr;
        pim_initialized = false;
    }
    PIM_PROFILE_TOCK(Deinitialize);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

PimBo* PimCreateBo(int w, int h, int c, int n, PimPrecision precision, PimMemType mem_type)
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
    size_t size = w * h * c * n * type_size;

    pim_bo->size = size;
    pim_bo->bshape = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
    pim_bo->bshape_r = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
    pim_bo->mem_type = mem_type;
    pim_bo->precision = precision;

    ret = pim_runtime->alloc_memory(pim_bo);
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

    if (!user_ptr) {
        ret = pim_runtime->alloc_memory(pim_bo);
        if (ret != 0) {
            DLOG(ERROR) << "Fail to alloc memory";

            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return nullptr;
        }

        pim_bo->use_user_ptr = false;
    } else {
        pim_bo->data = user_ptr;
        pim_bo->use_user_ptr = true;
    }

#ifdef EMULATOR
    pad_data(pim_bo->data, pim_desc, mem_type, mem_flag);
#endif

    PIM_PROFILE_TOCK(CreateBo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_bo;
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
    pim_desc->bshape_r = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};
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

    if (!pim_bo->use_user_ptr) {
        ret = pim_runtime->free_memory(pim_bo);
        if (ret != 0) {
            DLOG(ERROR) << "Fail to Destroy Bo";
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return -1;
        }
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

PimGemvBundle* PimGetGemvBundle(PimBo* weight, PimBo* dev_in, PimBo* dev_out)
{
    /*
    If the bundle already exists: return that bundle
    Else: create a new bundle, fill it up and return it
    */
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(GetGemvBundle);

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    PimDesc* pim_desc = PimCreateDesc(weight->bshape_r.n, 1, weight->bshape_r.h, weight->bshape_r.w, PIM_FP16, OP_GEMV);
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
        hipMemcpy(host_weight->data, weight->data, w_size * sizeof(half_float::half), hipMemcpyDeviceToHost);
    }

    // TODO: change the key from uint64_t to (keybo*, size): pull_req: 456)?
    uint64_t w_addr = reinterpret_cast<uint64_t>(weight->data);

    PimGemvBundle* bundle = nullptr;
    bundle = pim_runtime->find_gemv_bundle(w_addr);

    if (bundle == nullptr) {
        uint32_t w_size = pim_desc->bshape_r.h * pim_desc->bshape_r.w;
        PimBo* host_reordered_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
        PimBo* pre_wei = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);

        pim_runtime->convert_data_layout(host_reordered_weight, host_weight, OP_GEMV);
        PimCopyMemory(pre_wei, host_reordered_weight, HOST_TO_PIM);

        // bundle = PimCreateGemvBundle(dev_in, pre_wei, dev_out);
        bundle = new PimGemvBundle;
        bundle->in = dev_in;
        bundle->wei = pre_wei;
        bundle->out = dev_out;

        pim_runtime->insert_gemv_bundle(w_addr, bundle);

        PimDestroyDesc(pim_desc);
        PimDestroyBo(host_reordered_weight);
        if (host_weight != weight) {
            PimDestroyBo(host_weight);
        }
    }

    PIM_PROFILE_TOCK(GetGemvBundle);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return bundle;
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
    PimBo* padded_scalar = PimCreateBo(num_vector, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    for (int i = 0; i < num_vector; i++) {
        pim_runtime->copy_memory((void*)((char*)padded_scalar->data + i * sizeof(uint16_t)), scalar, sizeof(uint16_t),
                                 HOST_TO_PIM);
    }

    ret = pim_runtime->execute_add(output, vector, padded_scalar, stream, block);
    PIM_PROFILE_TOCK(ExecuteAdd);

    PimDestroyBo(padded_scalar);
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
    PimBo* padded_scalar = PimCreateBo(num_vector, 1, 1, 1, PIM_FP16, MEM_TYPE_PIM);
    for (int i = 0; i < num_vector; i++) {
        pim_runtime->copy_memory((void*)((char*)padded_scalar->data + i * sizeof(uint16_t)), scalar, sizeof(uint16_t),
                                 HOST_TO_PIM);
    }

    ret = pim_runtime->execute_mul(output, vector, padded_scalar, stream, block);
    PIM_PROFILE_TOCK(ExecuteMul);

    PimDestroyBo(padded_scalar);
    return ret;
}

int PimExecuteGemv(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    // Assuming operand1 is always weight, operand0 is always input
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(ExecuteGEMV);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    PimGemvBundle* bundle = PimGetGemvBundle(operand1, operand0, output);
    operand1 = bundle->wei;

    ret = pim_runtime->execute_gemv(output, operand0, operand1, stream, block);
    PIM_PROFILE_TOCK(ExecuteGEMV);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimExecuteGemvAdd(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    // Assuming operand1 is always weight, operand0 is always input
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PIM_PROFILE_TICK(ExecuteGEMVAdd);
    int ret = 0;

    if (pim_runtime == nullptr) {
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }

    PimGemvBundle* bundle = PimGetGemvBundle(operand1, operand0, output);
    operand1 = bundle->wei;

    ret = pim_runtime->execute_gemv_add(output, operand0, operand1, stream, block);
    PIM_PROFILE_TOCK(ExecuteGEMVAdd);

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
