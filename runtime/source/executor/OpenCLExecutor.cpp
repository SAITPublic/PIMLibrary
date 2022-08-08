/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "executor/OpenCLExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "CL/opencl.h"
#include "executor/PimExecutor.h"
#include "executor/pim_opencl_kernels/gpu_add_kernel.pimk"
#include "utility/assert_cl.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

namespace pim
{
namespace runtime
{
namespace executor
{
OpenCLExecutor::OpenCLExecutor(PimRuntimeType rt_type, PimPrecision precision) : PimExecutor(rt_type, precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

int OpenCLExecutor::initialize()
{
    int ret = PimExecutor::initialize();
    DLOG(INFO) << "[START] " << __FUNCTION__ << "OpenCL executor Intialization ";

    max_crf_size_ = 128;
    int max_srf_size = 2048;
    int zero = 0;

    pim_manager_->alloc_memory((void**)&d_srf_bin_buffer_, max_srf_size * 2, MEM_TYPE_DEVICE);
    pim_manager_->alloc_memory((void**)&zero_buffer_, 32 * 2, MEM_TYPE_DEVICE);
    auto cl_command_queue_ptr = static_cast<cl_command_queue>(pim_manager_->pim_memory_manager_->get_queue());
    clEnqueueFillBuffer(cl_command_queue_ptr, zero_buffer_, (void*)&zero, sizeof(int), 0, 32 * 2, 0, NULL, NULL);

    /* PIM HW can generate only gemv output without reduction sum */
    /* so PimExecutor needs to maintain intermediate output buffer for gemv op */
    pim_manager_->alloc_memory((void**)&pim_gemv_tmp_buffer_, 8 * 2 * 1024 * 1024, MEM_TYPE_PIM);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OpenCLExecutor::deinitialize(void)
{
    int ret = PimExecutor::deinitialize();
    DLOG(INFO) << " [START] " << __FUNCTION__ << " called";
    clReleaseMemObject(d_srf_bin_buffer_);
    clReleaseMemObject(zero_buffer_);
    pim_manager_->free_memory((void*)pim_gemv_tmp_buffer_, MEM_TYPE_PIM);
    // where is crf_lut allocated host or device. and why hipfree is used if host_Allocated.

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OpenCLExecutor::execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;
    cl_int exe_err;

    auto cl_context_ptr = static_cast<cl_context>(pim_manager_->pim_memory_manager_->get_context());
    auto cl_device_ptr = static_cast<cl_device_id*>(pim_manager_->pim_memory_manager_->get_device());
    cl_program program = clCreateProgramWithSource(cl_context_ptr, 1, (const char**)&source, NULL, &exe_err);
    cl_ok(exe_err);
    clBuildProgram(program, 1, cl_device_ptr, NULL, NULL, &exe_err);
    cl_ok(exe_err);
    cl_kernel kernel = clCreateKernel(program, "gpuAdd", &exe_err);
    cl_ok(exe_err);

    const size_t local_work_size = 32;
    const unsigned int length = operand0->size / 2;  // num floats.
    const size_t global_work_size = ceil(length / (float)local_work_size) * local_work_size;
    exe_err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&operand0->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&operand1->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&length);
    cl_ok(exe_err);

    auto cl_command_queue_ptr = static_cast<cl_command_queue>(pim_manager_->pim_memory_manager_->get_queue());
    exe_err = clEnqueueNDRangeKernel(cl_command_queue_ptr, kernel, 1, NULL, &global_work_size, &local_work_size, 0,
                                     NULL, NULL);
    cl_ok(exe_err);
    clFinish(cl_command_queue_ptr);
    return ret;
}
}  // namespace executor
}  // namespace runtime
}  // namespace pim
