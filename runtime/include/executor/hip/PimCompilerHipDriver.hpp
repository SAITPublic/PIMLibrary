/**
 * @file PimCompilerHipDriver.h
 * @brief Header file for PimCompilerDriver HIP implementation
 *
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or
 * computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung
 * Electronics.
 */
#ifndef __PIMC_DRIVER_HIP_HPP__
#define __PIMC_DRIVER_HIP_HPP__

#include <pim_runtime_api.h>
#include "executor/PimCompilerDriver.h"
#include "manager/HostInfo.h"

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
class HIPCompiler : public Compiler
{
   public:
    /**
     * @brief Constructor for HIPCompiler class
     *
     * @param pim_op pim_op object returned from compiler along with buffers from PIMLibrary
     */
    HIPCompiler() : Compiler() {}
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

class HIPDriver : public Driver
{
   public:
    HIPDriver() = default;
    HIPDriver(HIPDriver &&) = delete;
    HIPDriver(const HIPDriver &) = delete;
    HIPDriver &operator=(HIPDriver &&) = delete;
    HIPDriver &operator=(const HIPDriver &) = delete;

#if PIM_COMPILER_ENABLE == 1
    PimCompiledObj *build_program(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs,
                                  std::vector<PimBo *> input_pimbo, PimTarget *target, std::string compile_opts);
    PimBo *execute_program(PimCompiledObj *obj, PimTarget *target, std::string launch_opts);
#endif
    // todo:: Pass HW information from user

   private:
    hipFunction_t compile_code_hip(std::string kernel, std::string crf_binary);
    HIPCompiler compile_;
};
}  // namespace pimc_driver
}  // namespace runtime
}  // namespace pim
#endif
