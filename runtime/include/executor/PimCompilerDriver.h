/**
 * @file PimCompilerDriver.h
 * @brief Header file for PimCompilerDriver base implementation
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
#ifndef __PIMC_DRIVER_HPP__
#define __PIMC_DRIVER_HPP__

#include <pim_runtime_api.h>
#include "manager/HostInfo.h"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <memory>
#include <vector>

extern uint64_t g_pim_base_addr[MAX_NUM_GPUS];

namespace pim
{
namespace runtime
{
namespace pimc_driver
{
/**
 * Base class for defining host Compiler invoker
 *
 */
class Compiler
{
   public:
    /**
     * @brief Constructor for Compiler class
     * @param None
     */
    Compiler() = default;
    ~Compiler() {}
    /**
     * @brief Pure virtual function for invoking host compiler
     * @param host_code host code in string format
     * @param crf_binary generated CRF binary in string format
     * @return None
     */
    virtual void execute(std::string host_code, std::string crf_binary) = 0;

   private:
    Compiler(Compiler &&) = delete;
    Compiler(const Compiler &) = delete;
    Compiler &operator=(Compiler &&) = delete;
    Compiler &operator=(const Compiler &) = delete;
};
/**
 * Base class for PIM Compiler and device execution
 *
 */
class Driver
{
   public:
    /**
     * @brief Constructor for Driver class
     * @param None
     */
    Driver() = default;

#if PIM_COMPILER_ENABLE == 1
    /**
     * @brief Static Factory method for creating different drivers
     * based on runtime information
     * @param target PimTarget pointer containing runtime information
     * @return std::shared_ptr<Driver> pointer to created driver
     */
    static std::shared_ptr<Driver> create_driver(PimTarget *target);

    /**
     * @brief Pure virtual function for invoking PIMCompiler and generate
     * binary and host code
     * @param output Var designated as output
     * @param inputs vector of input buffers
     * @param input_pimbo vector of PIMLibrary input buffers
     * @param target PimTarget pointer containing target information
     * @param compile_opts compilation options
     * @return PimCompiledObj* pointer to generated PIMCompiler object
     */
    virtual PimCompiledObj *build_program(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs,
                                          std::vector<PimBo *> input_pimbo, PimTarget *target,
                                          std::string compile_opts) = 0;
    /**
     * @brief Pure virtual function for invoking host compiler and
     * executing generated code on target
     * @param obj generated PimCompilerObj
     * @param target PimTarget pointer containing target information
     * @param launch_opts execution options
     * @return PimBo* Output Pimbo
     */
    virtual PimBo *execute_program(PimCompiledObj *obj, PimTarget *target, std::string launch_opts) = 0;
#endif
    // todo:: Pass HW information from user

   private:
    Driver(Driver &&) = delete;
    Driver(const Driver &) = delete;
    Driver &operator=(Driver &&) = delete;
    Driver &operator=(const Driver &) = delete;
};
}  // namespace pimc_driver
}  // namespace runtime
}  // namespace pim
#endif
