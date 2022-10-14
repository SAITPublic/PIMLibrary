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
#include "executor/IPimExecutor.h"
#include "executor/PimExecutorFactory.h"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_util.h"
#include "executor/PimCompilerDriver.h"

using namespace pim::runtime::pimc_driver;

namespace pim
{
namespace runtime
{
PimRuntime::PimRuntime(PimRuntimeType rt_type, PimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    pim_manager_ = manager::PimManager::get_instance(rt_type, precision);
    pim_executor_ = executor::PimExecutorFactory::getPimExecutor(pim_manager_, this, rt_type, precision);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

PimRuntime::~PimRuntime(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    pim_executor_.reset();
    //    delete pim_manager_;
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
        free_memory(it->second);
        delete it->second;
    }

    pim_manager_->deinitialize();
    pim_executor_->deinitialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::set_device(uint32_t device_id)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    hipError_t deviceSet = hipSetDevice(device_id);
    if (hipSuccess != deviceSet) {
        DLOG(ERROR) << "Failed to set device " << deviceSet << "Device ID: " << device_id << std::endl;
        ret = int(deviceSet);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::get_device(uint32_t* device_id)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    hipError_t deviceGet = hipGetDevice((int*)device_id);
    if (hipSuccess != deviceGet) {
        DLOG(ERROR) << "Failed to get device " << deviceGet << "Device ID: " << device_id << std::endl;
        ret = int(deviceGet);
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

void* PimRuntime::createStream(PimRuntimeType rt_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    void* stream_obj = NULL;
    if (rt_type == RT_TYPE_HIP) {
        stream_obj = pim_executor_->createStream();
    } else {
        DLOG(ERROR) << __FUNCTION__ << " not implemented for runtime " << rt_type << " \n";
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return stream_obj;
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

    if (cpy_type == PIM_TO_PIM) {
        ret = pim_executor_->execute_copy(dst, src, NULL, true);
    } else {
        ret = pim_manager_->copy_memory(dst, src, cpy_type);
    }
    dst->data_layout_type = src->data_layout_type;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::copy_memory_3d(const PimCopy3D* copy_params)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_manager_->copy_memory_3d(copy_params);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = pim_executor_->execute_add(output, operand0, operand1, stream, block);

    return ret;
}

int PimRuntime::execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = pim_executor_->execute_mul(output, operand0, operand1, stream, block);

    return ret;
}

int PimRuntime::execute_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func, void* stream,
                             bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    ret = pim_executor_->execute_gemm(output, input, weight, bias, act_func, stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_relu(output, pim_data, stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                           double epsilon, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_bn(output, pim_data, beta, gamma, mean, variance, epsilon, stream, block);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimRuntime::execute_sync(void* stream)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = pim_executor_->execute_sync(stream);

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

PimBo* PimRuntime::find_preloaded_pim_weight(PimBo* weight)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PimBo* addr = nullptr;

    uint32_t w_key = 0;
    uint32_t* w_addr_ptr = reinterpret_cast<uint32_t*>(weight->data);
    int step = weight->size >> 1;

    for (int i = 0; i < weight->size / sizeof(uint32_t); i += step) {
        w_key ^= w_addr_ptr[i];
    }
    std::unordered_map<uint32_t, PimBo*>::const_iterator found = weight_map_.find(w_key);
    if (found != weight_map_.end()) {
        addr = found->second;
    } else {
        DLOG(INFO) << "[%s] not found\tw_addr:%p, w_key:%X, weight_map_size:%d\n"
                   << __func__ << weight->data << w_key << weight_map_.size();
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return addr;
}

int PimRuntime::insert_preloaded_pim_weight(PimBo* dev_wei, PimBo* pim_wei)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    uint32_t w_key = 0;
    uint32_t* w_addr_ptr = reinterpret_cast<uint32_t*>(dev_wei->data);
    int step = dev_wei->size >> 1;
    for (int i = 0; i < dev_wei->size / sizeof(uint32_t); i += step) {
        w_key ^= w_addr_ptr[i];
    }
    weight_map_.insert(std::make_pair(w_key, pim_wei));
    DLOG(INFO) << "[%s] insert\tw_addr:%p, w_key:%X, weight_map_size:%lu\n"
               << __func__ << dev_wei->data << w_key << weight_map_.size();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

PimBo* PimRuntime::get_preloaded_pim_gemm_weight(PimBo* dev_wei, bool save_for_reuse)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PimBo* pre_wei = find_preloaded_pim_weight(dev_wei);

    if (pre_wei == nullptr) {
        PimBo* host_weight = nullptr;
        PimBo* host_weight_t = nullptr;
        PimBo* host_reordered_weight = nullptr;
        PimBShape* bshape = &dev_wei->bshape;

        if (dev_wei->data == nullptr) {
            DLOG(ERROR) << "[END] " << __FUNCTION__ << " called";
            return nullptr;
        }
        uint32_t w_size = dev_wei->size_r;
        host_weight = PimCreateBo(bshape->n, bshape->c, bshape->h, bshape->w, PIM_FP16, MEM_TYPE_HOST);
        host_weight_t = PimCreateBo(bshape->n, bshape->c, bshape->h, bshape->w, PIM_FP16, MEM_TYPE_HOST);

        pim_manager_->copy_memory(host_weight_t, dev_wei, DEVICE_TO_HOST);
        transpose_pimbo(host_weight, host_weight_t);

        host_reordered_weight = PimCreateBo(bshape->n, bshape->c, bshape->h, bshape->w, PIM_FP16, MEM_TYPE_HOST);
        pre_wei = PimCreateBo(bshape->n, bshape->c, bshape->h, bshape->w, PIM_FP16, MEM_TYPE_PIM);
        pim_manager_->convert_data_layout(host_reordered_weight, host_weight);
        PimCopyMemory(pre_wei, host_reordered_weight, HOST_TO_PIM);
        if (save_for_reuse) {
            insert_preloaded_pim_weight(dev_wei, pre_wei);
        }
        std::cout<<"PimDestryBo called in "<<__FUNCTION__<<std::endl;
        PimDestroyBo(host_reordered_weight);

        if (host_weight != dev_wei) {
            std::cout<<"PimDestryBo called in "<<__FUNCTION__<<std::endl;
            PimDestroyBo(host_weight);
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pre_wei;
}

PimBo* PimRuntime::generate_gemm_weight_from_buffer(PimBo* src, bool save_for_reuse)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    PimBo* pre_weight = nullptr;
    if (src->data_layout_type == PimDataLayoutType::RAW && is_pim_applicable(src)) {
        pre_weight = get_preloaded_pim_gemm_weight(src, save_for_reuse);
    } else {
        DLOG(ERROR) << "GEMM weight generation for provided layout is not supported yet";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }

    PimBo* pim_reordered_buff;
    // Make sure reordered data are stored on the PIM memory
    if (pre_weight->mem_type == MEM_TYPE_PIM) {
        pim_reordered_buff = pre_weight;
    } else {
        const auto direction = pre_weight->mem_type == MEM_TYPE_HOST ? HOST_TO_PIM : DEVICE_TO_PIM;
        pim_reordered_buff =
            PimCreateBo(pre_weight->bshape.n, pre_weight->bshape.c, pre_weight->bshape.h, pre_weight->bshape.w,
                        pre_weight->precision, MEM_TYPE_PIM, nullptr, pre_weight->transposed);
        PimCopyMemory(pim_reordered_buff, pre_weight, direction);
        if (pre_weight != src) {
            std::cout<<"PimDestryBo called in "<<__FUNCTION__<<std::endl;
            PimDestroyBo(pre_weight);
        }
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_reordered_buff;
}

PimCompiledObj* PimRuntime::build_program(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs, std::vector<PimBo*> input_pimbo, PimTarget* target, std::string compile_opts)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    //HIP implementation
    //Create input PimBo map

    std::vector<PimBo*> new_pimbo;
    PimBo* output_pimbo;
    std::unordered_map<std::string, PimBo*> pimbo_map;
    if (inputs.size() != input_pimbo.size()) {
        DLOG(ERROR) << "Number of input Buffers and PimBos are not same";
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return nullptr;
    }
    for (int i = 0; i < inputs.size(); i++) {
        pimbo_map[inputs[i].get_name()] = input_pimbo[i];
    }

    //Create output pimbo
    auto indices = output.get_indices();
    int n = 1;
    int c = 1;
    int h = 1;
    int w = 1;
    uint32_t num = indices.size();
    if (num > 0)
        w = indices[num - 1]->get_stop() - indices[num - 1]->get_start();
    if (num > 1)
        h = indices[num - 2]->get_stop() - indices[num - 2]->get_start();
    if (num > 2)
        c = indices[num - 3]->get_stop() - indices[num - 3]->get_start();
    if (num > 3)
        n = indices[num - 4]->get_stop() - indices[num - 4]->get_start();
    output_pimbo = PimCreateBo(n, c, h, w, PimPrecision::PIM_FP16, PimMemType::MEM_TYPE_PIM);
    pimbo_map[output.name()] = output_pimbo;

    //Compile code
    output.generate_code(); //can be added to below function
    auto compiled_obj = pimc::get_compiled_object(output, compile_opts);
    //Create temp PimBos
    for (auto buf : compiled_obj->get_extra_buffers()) {
        auto pimbo = PimCreateBo(1, 1, 1, buf->size(), PimPrecision::PIM_FP16, PimMemType::MEM_TYPE_PIM);
        pimbo_map[buf->name()] = pimbo;
        new_pimbo.push_back(pimbo);
    }

    //TODO Reorder weights

    //Wrap in PimCompiledObj
    PimCompiledObj *pim_co = new PimCompiledObj;
    pim_co->output_pimbo = output_pimbo;
    pim_co->input_pimbo = input_pimbo;
    pim_co->new_pimbo = new_pimbo;
    pim_co->kernel = compiled_obj->get_gpu_kernel();
    pim_co->crf_binary = compiled_obj->get_crf_binary();
    pim_co->num_blocks = compiled_obj->get_number_of_blocks();
    pim_co->num_threads = compiled_obj->get_number_of_threads();
    pim_co->op_order = compiled_obj->get_op_order();
    pim_co->pimbo_map = pimbo_map;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return pim_co;
}

PimBo* PimRuntime::execute_program(PimCompiledObj* obj, PimTarget* target, std::string launch_opts) {
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    pimc_driver::PimCDriver pimc_driver;
    auto kernel = pimc_driver.compile_code(obj->kernel, obj->crf_binary);
    size_t num_args = obj->op_order.size() + 1;
    uint8_t* args[num_args];// +1 for gpim_base_addr
    uint8_t *crf_binary_device;
    hipMalloc((void **)&crf_binary_device, 128);
    hipMemcpy((void *)crf_binary_device, (uint8_t *)(obj->crf_binary.c_str()), obj->crf_binary.size(), hipMemcpyHostToDevice);
    for (size_t i = 0; i < obj->op_order.size(); i++) {
        if (obj->pimbo_map.find(obj->op_order[i]) != obj->pimbo_map.end()){
            args[i] = static_cast<uint8_t*>(obj->pimbo_map[obj->op_order[i]]->data);

        }else if (obj->op_order[i] == "crf_binary") {
            //Push pim_ctr
            args[i++] = (uint8_t *)g_pim_base_addr[0];
            args[i] = crf_binary_device;
        }
        else {
            DLOG(ERROR) << "PimBo not found in map";
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return nullptr;
        }
    }
    void *config[5] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, nullptr, HIP_LAUNCH_PARAM_BUFFER_SIZE, &num_args, HIP_LAUNCH_PARAM_END};
    config[1] = static_cast<void*>(args);
    hipModuleLaunchKernel(kernel, obj->num_blocks, 1, 1, obj->num_threads, 1, 1, 0, nullptr, NULL, reinterpret_cast<void **>(&config));

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return obj->output_pimbo;
}

} /* namespace runtime */
} /* namespace pim */
