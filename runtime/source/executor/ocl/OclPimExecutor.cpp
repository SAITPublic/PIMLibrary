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

namespace pim
{
namespace runtime
{
extern cl_platform_id platform;
extern cl_context context;
extern cl_device_id device_id;
extern cl_command_queue queue;

namespace executor
{
OclPimExecutor::OclPimExecutor(pim::runtime::manager::PimManager* pim_manager, PimPrecision precision)
    : pim_manager_(pim_manager), precision_(precision), max_crf_size_(128)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";

    int ret = 0;

    ret = check_cl_program_path();
    if (ret == 0 /* source path */) {
        build_cl_program_with_source();
        save_cl_program_binary();
    } else if (ret == 1 /* binary path */) {
        build_cl_program_with_binary();
    } else {
        assert(0);
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

OclPimExecutor::~OclPimExecutor(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";
    pim_device_.reset();
}

int OclPimExecutor::check_cl_program_path(void)
{
    cl_binary_path_ = CL_KERNEL_BINARY_PATH;
    cl_binary_path_ += "ocl_pimk.bin";
    std::ifstream cl_binary_file(cl_binary_path_, std::ios::in);

    if (cl_binary_file.is_open()) {
        // printf("cl binary is available\n");
        cl_binary_file.close();
        return 1;
    }

    std::string cl_source_path;
    cl_source_path = CL_KERNEL_SOURCE_PATH;
    cl_source_path += "pim_op_kernels.cl";
    std::ifstream cl_source_file(cl_source_path.c_str(), std::ios::in);

    if (cl_source_file.is_open()) {
        // printf("cl source is available\n");
        cl_source_file.close();
        return 0;
    }

    return -1;
}

std::string OclPimExecutor::load_cl_file(std::string filename)
{
    std::string result;

    std::string cl_source_path = CL_KERNEL_SOURCE_PATH;
    cl_source_path += filename;
    std::ifstream cl_source_file(cl_source_path.c_str(), std::ios::in);
    std::ostringstream oss;

    oss << cl_source_file.rdbuf();
    result = oss.str();

    cl_source_file.close();

    return result;
}

int OclPimExecutor::build_cl_program_with_source(void)
{
    int ret = 0;
    std::string cl_source;
    const char* cl_source_ptr = nullptr;

    cl_source = load_cl_file("pim_op_kernels.cl");
    cl_source += load_cl_file("gpu_op_kernels.cl");
    cl_source_ptr = cl_source.c_str();

    program_ = clCreateProgramWithSource(context, 1, (const char**)&cl_source_ptr, NULL, NULL);
    clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);

    return ret;
}

int OclPimExecutor::save_cl_program_binary(void)
{
    int ret = 0;
    cl_uint num_devices = 0;
    std::string cl_binary_name = "ocl_pimk.bin";
    const char* cl_binary_ptr = nullptr;
    size_t cl_binary_size = 0;

    clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &cl_binary_size, NULL);
    cl_binary_ptr = new char[cl_binary_size];
    clGetProgramInfo(program_, CL_PROGRAM_BINARIES, sizeof(char*), (char**)&cl_binary_ptr, NULL);

    ofstream bf(cl_binary_name.c_str());
    bf.write(cl_binary_ptr, cl_binary_size);

    bf.close();
    delete[] cl_binary_ptr;

    return ret;
}

int OclPimExecutor::build_cl_program_with_binary(void)
{
    int ret = 0;
    std::ifstream bf(cl_binary_path_.c_str(), ios::binary);
    unsigned char* cl_binary_ptr = nullptr;
    size_t cl_binary_size = 0;

    bf.seekg(0, ios::end);
    cl_binary_size = bf.tellg();
    bf.seekg(0, ios::beg);
    cl_binary_ptr = new unsigned char[cl_binary_size];
    bf.read((char*)cl_binary_ptr, cl_binary_size);

    program_ = clCreateProgramWithBinary(context, 1, &device_id, &cl_binary_size, (const unsigned char**)&cl_binary_ptr,
                                         NULL, NULL);
    clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);

    delete[] cl_binary_ptr;

    return ret;
}

int OclPimExecutor::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << "OpenCL executor Intialization ";
    int ret = 0;
    int max_srf_size = 2048;
    int zero = 0;

    pim_manager_->alloc_memory((void**)&d_srf_bin_buffer_, max_srf_size * 2, MEM_TYPE_DEVICE);
    pim_manager_->alloc_memory((void**)&zero_buffer_, 32 * 2, MEM_TYPE_DEVICE);
    clEnqueueFillBuffer(queue, zero_buffer_, (void*)&zero, sizeof(int), 0, 32 * 2, 0, NULL, NULL);

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

    cl_kernel kernel = clCreateKernel(program_, "pim_add_test", NULL);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&operand0->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&operand1->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output->data);
    cl_ok(exe_err);
    exe_err = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&length);
    cl_ok(exe_err);

    exe_err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    cl_ok(exe_err);
    clFinish(queue);
    return ret;
}
}  // namespace executor
}  // namespace runtime
}  // namespace pim
