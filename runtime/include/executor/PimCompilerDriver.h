/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */
#ifndef __PIMC_DRIVER_H__
#define __PIMC_DRIVER_H__

#include <pim_runtime_api.h>
#include "manager/HostInfo.h"
//#include <pim_compiler.hpp>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <vector>

extern uint64_t g_pim_base_addr[MAX_NUM_GPUS];

namespace pim
{
namespace runtime
{
namespace pimc_driver
{
class HIPCompiler
{
   public:
    /**
     * @brief Constructor for HIPCompiler class
     *
     * @param pim_op pim_op object returned from compiler along with buffers from PIMLibrary
     */
    HIPCompiler() {}
    ~HIPCompiler() {}
    void execute(std::string hip_kernel, std::string crf_binary);
    hipFunction_t get_kernel_function() { return kernel_; }

   private:
    HIPCompiler(HIPCompiler &&) = delete;
    HIPCompiler(const HIPCompiler &) = delete;
    HIPCompiler &operator=(HIPCompiler &&) = delete;
    HIPCompiler &operator=(const HIPCompiler &) = delete;

#if PIM_COMPILER_ENABLE == 1
    hipModule_t module_;
#endif
    hipFunction_t kernel_;
};

class PimCDriver
{
   public:
    PimCDriver() = default;
    PimCDriver(PimCDriver &&) = delete;
    PimCDriver(const PimCDriver &) = delete;
    PimCDriver &operator=(PimCDriver &&) = delete;
    PimCDriver &operator=(const PimCDriver &) = delete;

#if PIM_COMPILER_ENABLE == 1
    hipFunction_t compile_code_hip(std::string kernel, std::string crf_binary);
    PimCompiledObj *build_program(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs,
                                  std::vector<PimBo *> input_pimbo, PimTarget *target, std::string compile_opts);
    PimBo *execute_program(PimCompiledObj *obj, PimTarget *target, std::string launch_opts);
    void pim_malloc(void **ptr, size_t size, PimTarget *target);
    void pim_memcpy(void *dest, const void *src, size_t size, PimTarget *target);
    void pim_launch_kernel(std::string kernel, std::string crf_binary, uint32_t num_blocks, uint32_t num_threads,
                           uint8_t *args[], size_t num_args, PimTarget *target);
#endif
    // todo:: Pass HW information from user

   private:
    HIPCompiler compile_;
};
}  // namespace pimc_driver
}  // namespace runtime
}  // namespace pim
#endif
