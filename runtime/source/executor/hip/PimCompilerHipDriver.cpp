/**
 * @file PimCompilerHipDriver.cpp
 * @brief Source file for PimCompilerDriver HIP implementation
 *
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system, or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "executor/hip/PimCompilerHipDriver.hpp"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

namespace pim
{
namespace runtime
{
namespace pimc_driver
{
#if PIM_COMPILER_ENABLE == 1
void HIPCompiler::execute(std::string hip_kernel, std::string crf_binary)
{
    if (const char* env_p = std::getenv("SAVE_GENERATED_KERNEL")) {
        int value = *((int*)env_p);
        if (value == 1) {
            std::ofstream hip_file("output_hip_kernel.txt");
            hip_file << hip_kernel;
            hip_file.close();

            std::ofstream crf_file("output_crf_binary.txt");
            crf_file << crf_binary;
            crf_file.close();
        }
    }
    std::string compile_opts = "-O3 --gpu-architecture=gfx906";
    const char* options[] = {compile_opts.c_str()};

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, hip_kernel.c_str(), "generated_pim_source.cu", 0, nullptr, nullptr);
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);

    if (logSize) {
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);

        DLOG(INFO) << log;
    }

    if (compileResult != HIPRTC_SUCCESS)
        DLOG(ERROR) << "Compilation failed." << std::endl;
    else
        DLOG(INFO) << "Compilation successful" << std::endl;

    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);

    std::vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());

    hipError_t err;
    hiprtcDestroyProgram(&prog);
    err = hipModuleLoadData(&module_, code.data());
    if (err != hipSuccess) {
        DLOG(INFO) << "Falied to Load Module Err: " << err << std::endl;
    }
    err = hipModuleGetFunction(&kernel_, module_, "pim_hip_kernel");
    if (err != hipSuccess) {
        DLOG(INFO) << "Falied to Load function Err: " << err << std::endl;
    }
}

hipFunction_t HIPDriver::compile_code_hip(std::string kernel, std::string crf_binary)
{
    compile_.execute(kernel, crf_binary);
    return compile_.get_kernel_function();
}

PimCompiledObj* HIPDriver::build_program(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs,
                                         std::vector<PimBo*> input_pimbo, PimTarget* target, std::string compile_opts)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    // Create input PimBo map
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

    // Compile code
    output.generate_code();  // can be added to below function
    auto compiled_obj = pimc::get_compiled_object(output, compile_opts);

    // Create output pimbo
    auto indices = output.get_indices();
    int n = 1;
    int c = 1;
    int h = 1;
    int w = 1;
    uint32_t num = indices.size();
    if (num > 0) w = indices[num - 1]->get_stop() - indices[num - 1]->get_start();
    if (num > 1) h = indices[num - 2]->get_stop() - indices[num - 2]->get_start();
    if (num > 2) c = indices[num - 3]->get_stop() - indices[num - 3]->get_start();
    if (num > 3) n = indices[num - 4]->get_stop() - indices[num - 4]->get_start();
    auto output_loc = compiled_obj->get_output_targets();
    if (output_loc.find(output.name()) != output_loc.end() && output_loc[output.name()] == "device") {
        output_pimbo = PimCreateBo(n, c, h, w, PIM_FP16, PimMemType::MEM_TYPE_DEVICE);
    } else {
        output_pimbo = PimCreateBo(n, c, h, w, PIM_FP16, PimMemType::MEM_TYPE_PIM);
    }
    pimbo_map[output.name()] = output_pimbo;

    // Create temporary PimBos
    for (auto buf : compiled_obj->get_extra_buffers()) {
        auto pimbo = PimCreateBo(1, 1, 1, buf->size(), PimPrecision::PIM_FP16, PimMemType::MEM_TYPE_PIM);
        hipMemset(pimbo->data, 0x00, buf->size());
        pimbo_map[buf->name()] = pimbo;
        new_pimbo.push_back(pimbo);
    }

    // Reorder pimbos
    for (auto buf : compiled_obj->get_reorder_buffers()) {
        size_t index = 0;
        for (index = 0; index < inputs.size(); index++) {
            if (inputs[index].get_name() == buf) {
                break;
            }
        }
        assert(index < inputs.size());
        auto* reordered_pim_w = PimConvertGemmWeight(input_pimbo[index], I_X_W);
        pimbo_map[inputs[index].get_name()] = reordered_pim_w;
        input_pimbo[index] = reordered_pim_w;
    }

    // Wrap in PimCompiledObj
    PimCompiledObj* pim_co = new PimCompiledObj;
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

PimBo* HIPDriver::execute_program(PimCompiledObj* obj, PimTarget* target, std::string launch_opts)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    size_t num_args = obj->op_order.size() + 1;
    uint8_t* args[num_args];  // +1 for gpim_base_addr

    // Copy CRF binary
    uint8_t* crf_binary_device;
    hipMalloc((void**)&crf_binary_device, 128);
    hipMemcpy((void*)crf_binary_device, (uint8_t*)(obj->crf_binary.c_str()), obj->crf_binary.size(),
              hipMemcpyHostToDevice);

    // Setup args
    for (size_t i = 0; i < obj->op_order.size(); i++) {
        if (obj->pimbo_map.find(obj->op_order[i]) != obj->pimbo_map.end()) {
            args[i] = static_cast<uint8_t*>(obj->pimbo_map[obj->op_order[i]]->data);

        } else if (obj->op_order[i] == "crf_binary") {
            // Push pim_ctr
            args[i++] = (uint8_t*)g_pim_base_addr[3]; // Use GPU 4
            args[i] = crf_binary_device;
        } else {
            DLOG(ERROR) << "PimBo not found in map";
            DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
            return nullptr;
        }
    }

    // Launch kernel
    void* config[5] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, nullptr, HIP_LAUNCH_PARAM_BUFFER_SIZE, &num_args,
                       HIP_LAUNCH_PARAM_END};
    config[1] = static_cast<void*>(args);
    auto hip_kernel = compile_code_hip(obj->kernel, obj->crf_binary);
    hipModuleLaunchKernel(hip_kernel, obj->num_blocks, 1, 1, 64 /*FIXME */, 1, 1, 0, nullptr, NULL,
                          reinterpret_cast<void**>(&config));

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return obj->output_pimbo;
}
#endif
}  // namespace pimc_driver
}  // namespace runtime
}  // namespace pim
