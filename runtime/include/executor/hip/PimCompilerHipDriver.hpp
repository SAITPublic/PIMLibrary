/**
 * @file PimCompilerHipDriver.hpp
 * @brief Header file for PimCompilerDriver HIP implementation
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
/**
 * class for defining host HIP Compiler invoker
 */
class HIPCompiler : public Compiler
{
   public:
    /**
     * @brief Constructor for HIPCompiler class
     * @param None
     */
    HIPCompiler() : Compiler() {}
    ~HIPCompiler() {}
    /**
     * @brief Function for invoking host HIP compiler
     * @param host_code host code in string format
     * @param crf_binary generated CRF binary in string format
     * @return None
     */
    void execute(std::string hip_kernel, std::string crf_binary);

    /**
     * @brief Function to get generated kernel function
     * @param None
     * @return hipFunction_t generated HIP Function to be passed to HIP launch kernel
     */
    hipFunction_t get_kernel_function() { return kernel_; }
   private:
    HIPCompiler(HIPCompiler &&) = delete;
    HIPCompiler(const HIPCompiler &) = delete;
    HIPCompiler &operator=(HIPCompiler &&) = delete;
    HIPCompiler &operator=(const HIPCompiler &) = delete;

#if PIM_COMPILER_ENABLE == 1
    hipModule_t module_;    ///< Intermediate generated HIP module
#endif
    hipFunction_t kernel_;  ///< Generated HIP Function
};

class HIPDriver : public Driver
{
   public:
    HIPDriver() = default;

#if PIM_COMPILER_ENABLE == 1
    /**
     * @brief function for invoking PIMCompiler and generate
     * binary and host HIP code
     * @param output Var designated as output
     * @param inputs vector of input buffers
     * @param input_pimbo vector of PIMLibrary input buffers
     * @param target PimTarget pointer containing target information
     * @param compile_opts compilation options
     * @return PimCompiledObj* pointer to generated PIMCompiler object
     */
    PimCompiledObj *build_program(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs,
                                  std::vector<PimBo *> input_pimbo, PimTarget *target, std::string compile_opts);
    /**
     * @brief Function for invoking host HIP compiler and
     * executing generated code on target
     * @param obj generated PimCompilerObj
     * @param target PimTarget pointer containing target information
     * @param launch_opts execution options
     * @return PimBo* Output Pimbo
     */
    PimBo *execute_program(PimCompiledObj *obj, PimTarget *target, std::string launch_opts);
#endif

   private:
    hipFunction_t compile_code_hip(std::string kernel, std::string crf_binary);

    HIPDriver(HIPDriver &&) = delete;
    HIPDriver(const HIPDriver &) = delete;
    HIPDriver &operator=(HIPDriver &&) = delete;
    HIPDriver &operator=(const HIPDriver &) = delete;

    HIPCompiler compile_; ///< HIPCompiler driver object
};
}  // namespace pimc_driver
}  // namespace runtime
}  // namespace pim
#endif
