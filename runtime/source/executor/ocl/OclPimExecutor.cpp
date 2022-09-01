/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "executor/ocl/OclPimExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "utility/assert_cl.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

const char* source =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable              \n"
    "__kernel void gpuAdd(__global half* a,                     \n"
    "                     __global half* b,                     \n"
    "                     __global half* output,                \n"
    "                     const unsigned int n)                 \n"
    "{                                                          \n"
    "   int gid = get_global_id(0);                             \n"
    "   if(gid < n)                                             \n"
    "       output[gid] = a[gid] + b[gid];                      \n"
    "}                                                          \n";

namespace pim
{
namespace runtime
{
namespace executor
{
OclPimExecutor::OclPimExecutor(pim::runtime::manager::PimManager* pim_manager, PimPrecision precision)
    : pim_manager_(pim_manager), precision_(precision), max_crf_size_(128)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";

    clGetPlatformIDs(1, &platform_, NULL);
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_id_, NULL);
    context_ = clCreateContext(NULL, 1, &device_id_, NULL, NULL, NULL);
    queue_ = clCreateCommandQueue(context_, device_id_, 0, NULL);
    program_ = clCreateProgramWithSource(context_, 1, (const char**)&source, NULL, NULL);
    clBuildProgram(program_, 1, &device_id_, NULL, NULL, NULL);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

OclPimExecutor::~OclPimExecutor(void) {}

int OclPimExecutor::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << "OpenCL executor Intialization ";
    int ret = 0;
    int max_srf_size = 2048;
    int zero = 0;

    pim_manager_->alloc_memory((void**)&d_srf_bin_buffer_, max_srf_size * 2, MEM_TYPE_DEVICE);
    pim_manager_->alloc_memory((void**)&zero_buffer_, 32 * 2, MEM_TYPE_DEVICE);
    clEnqueueFillBuffer(queue_, zero_buffer_, (void*)&zero, sizeof(int), 0, 32 * 2, 0, NULL, NULL);

    /* PIM HW can generate only gemv output without reduction sum */
    /* so PimExecutor needs to maintain intermediate output buffer for gemv op */
    // pim_manager_->alloc_memory((void**)&pim_gemv_tmp_buffer_, 8 * 2 * 1024 * 1024, MEM_TYPE_PIM);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclPimExecutor::deinitialize(void)
{
    DLOG(INFO) << " [START] " << __FUNCTION__ << " called";
    int ret = 0;
    clReleaseMemObject(d_srf_bin_buffer_);
    clReleaseMemObject(zero_buffer_);
    // pim_manager_->free_memory((void*)pim_gemv_tmp_buffer_, MEM_TYPE_PIM);
    // where is crf_lut allocated host or device. and why hipfree is used if host_Allocated.

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclPimExecutor::execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;
    const size_t local_work_size = 32;
    const unsigned int length = operand0->size / 2;  // num floats.
    const size_t global_work_size = ceil(length / (float)local_work_size) * local_work_size;
    cl_uint exe_err = 0;

    cl_kernel kernel = clCreateKernel(program_, "gpuAdd", NULL);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&operand0->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&operand1->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&length);
    cl_ok(exe_err);

    exe_err = clEnqueueNDRangeKernel(queue_, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    cl_ok(exe_err);
    clFinish(queue_);
    return ret;
}
}  // namespace executor
}  // namespace runtime
}  // namespace pim
