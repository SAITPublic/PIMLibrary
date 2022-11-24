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

#include "executor/ocl/OclPimExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "manager/HostInfo.h"
#include "utility/assert_cl.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

extern uint64_t g_pim_base_addr[MAX_NUM_GPUS];

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
OclPimExecutor::OclPimExecutor(pim::runtime::manager::PimManager* pim_manager, pim::runtime::PimRuntime* pim_runtime,
                               PimPrecision precision)
    : pim_manager_(pim_manager), pim_runtime_(pim_runtime)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";

    int ret = 0;
    const char* env_k = std::getenv("PIM_KERNEL_TYPE");
    if (env_k != nullptr) {
        switch (*env_k) {
            case '1':
                kernel_type_ = PIM;
                break;
            case '2':
                kernel_type_ = CUSTOM_GPU;
                break;
            default:
                kernel_type_ = OPTIMAL;
        }
    }

    ret = check_cl_program_path();
    if (ret == 0 /* source path */) {
        ret = build_cl_program_with_source();
        if (ret != CL_SUCCESS) {
            DLOG(ERROR) << "Failed to build Pim Kernels";
            assert(0);
        }
        save_cl_program_binary();
    } else if (ret == 1 /* binary path */) {
        build_cl_program_with_binary();
    } else {
        assert(0);
    }

    pim_crf_generator_ = std::make_shared<PimCrfBinGen>(pim_manager_);
    pim_device_ = pim_manager_->get_pim_device();
    pbi_ = pim_device_->get_pim_block_info();
    base_address_ = nullptr;
    pim_gemv_type_ = TILE_ACCUM;

    eltwise_kernel_ = clCreateKernel(program_, "elt_op_pim", &exec_err_);
    cl_ok(exec_err_);
    relu_kernel_ = clCreateKernel(program_, "relu_pim_operation", &exec_err_);
    cl_ok(exec_err_);
    copy_kernel_ = clCreateKernel(program_, "copy_pim", &exec_err_);
    cl_ok(exec_err_);
    bn_kernel_ = clCreateKernel(program_, "bn_pim_nr_sip", &exec_err_);
    cl_ok(exec_err_);
    pim_aligned_gemm_bias_relu_8tile_fp16_ =
        clCreateKernel(program_, "pim_aligned_gemm_bias_relu_8tile_fp16", &exec_err_);
    cl_ok(exec_err_);

    pim_aligned_gemm_bias_relu_fp16_ = clCreateKernel(program_, "pim_aligned_gemm_bias_relu_fp16", &exec_err_);
    cl_ok(exec_err_);

    pim_chwise_gemm_bias_relu_32tile_fp16_ =
        clCreateKernel(program_, "pim_chwise_gemm_bias_relu_32tile_fp16", &exec_err_);
    cl_ok(exec_err_);

    pim_chwise_gemm_bias_relu_fp16_ = clCreateKernel(program_, "pim_chwise_gemm_bias_relu_fp16", &exec_err_);
    cl_ok(exec_err_);
#ifdef EMULATOR
    pim_emulator_ = std::make_shared<pim::runtime::emulator::OclPimEmulator>();
    fmtd_size_per_ch_ = 100000;
    max_block_size_ = pbi_->num_pim_chan;
    max_fmtd_size_ = fmtd_size_per_ch_ * max_block_size_;
#endif
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

OclPimExecutor::~OclPimExecutor(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";
    pim_device_.reset();
    clReleaseKernel(eltwise_kernel_);
    clReleaseKernel(relu_kernel_);
    clReleaseKernel(copy_kernel_);
    clReleaseKernel(bn_kernel_);
    clReleaseKernel(pim_aligned_gemm_bias_relu_8tile_fp16_);
    clReleaseKernel(pim_aligned_gemm_bias_relu_fp16_);
    clReleaseKernel(pim_chwise_gemm_bias_relu_32tile_fp16_);
    clReleaseKernel(pim_chwise_gemm_bias_relu_fp16_);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called ";
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

    cl_source = load_cl_file("PimInfo.cl");
    cl_source += load_cl_file("kernel_utils.cl");
    cl_source += load_cl_file("pim_op_kernels.cl");
    cl_source += load_cl_file("pim_gemm.cl");
    cl_source += load_cl_file("pim_copy.cl");
    cl_source += load_cl_file("pim_bn.cl");
    cl_source_ptr = cl_source.c_str();

    program_ = clCreateProgramWithSource(context, 1, (const char**)&cl_source_ptr, NULL, NULL);
    string options = "-I" + std::string(CL_KERNEL_INCLUDE_PATH) + "/manager";
#ifdef EMULATOR
    options += " -DEMULATOR";
#endif
    // to disable the kernel opt while compliling binary so that it does not remove the dummy read calls meant for PIM.
    options += " -cl-opt-disable";

    ret = clBuildProgram(program_, 1, &device_id, options.c_str(), NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t len;
        cl_build_status bldstatus;

        printf("\nError %d: Failed to build program executable [ %s ]\n", ret, clGetErrorString(ret));
        ret = clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void*)&bldstatus,
                                    &len);
        printf("INFO: %s\n", clGetErrorString(bldstatus));
        ret = clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char* buffer = new char[len];

        ret = clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_OPTIONS, len, buffer, NULL);
        printf("Build Options %d: %s\n", ret, clGetErrorString(ret));
        printf("INFO: %s\n", buffer);
        ret = clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("Build Log %d: %s\n", ret, clGetErrorString(ret));
        printf("%s\n", buffer);

        delete[] buffer;
    }

    return ret;
}

int OclPimExecutor::save_cl_program_binary(void)
{
    int ret = 0;
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
    ret = clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t len;
        cl_build_status bldstatus;

        printf("\nError %d: Failed to build program executable [ %s ]\n", ret, clGetErrorString(ret));
        ret = clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void*)&bldstatus,
                                    &len);
        printf("INFO: %s\n", clGetErrorString(bldstatus));
        ret = clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char* buffer = new char[len];

        ret = clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_OPTIONS, len, buffer, NULL);
        printf("Build Options %d: %s\n", ret, clGetErrorString(ret));
        printf("INFO: %s\n", buffer);
        ret = clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("Build Log %d: %s\n", ret, clGetErrorString(ret));
        printf("%s\n", buffer);

        delete[] buffer;
    }
    delete[] cl_binary_ptr;

    return ret;
}

int OclPimExecutor::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << "OpenCL executor Intialization ";
    int ret = 0;
    int max_srf_size = 2048;
    int zero = 0;
    int is_gemv_tile_tree = pim_gemv_type_ == TILE_TREE ? 1 : 0;
    pim_crf_generator_->set_gemv_tile_tree(is_gemv_tile_tree);

    pim_manager_->alloc_memory((void**)&d_srf_bin_buffer_, max_srf_size * 2, MEM_TYPE_DEVICE);
    pim_manager_->alloc_memory((void**)&zero_buffer_, 32 * 2, MEM_TYPE_DEVICE);
    clEnqueueFillBuffer(queue, (cl_mem)zero_buffer_, (void*)&zero, sizeof(int), 0, 32 * 2, 0, NULL, NULL);

    /* PIM HW can generate only gemv output without reduction sum */
    /* so PimExecutor needs to maintain intermediate output buffer for gemv op */
    pim_manager_->alloc_memory((void**)&pim_gemv_tmp_buffer_, 8 * 2 * 1024 * 1024, MEM_TYPE_PIM);
    if (!base_address_) base_address_ = pim_manager_->get_pim_manager()->get_base_memobj();

#ifdef EMULATOR
    size_t reserved_fmtd_size = max_fmtd_size_ * sizeof(PimMemTraceData);

    d_emulator_trace_ = (PimMemTracer*)malloc(sizeof(PimMemTracer));
    h_fmtd16_ = (PimMemTraceData*)malloc(reserved_fmtd_size);
    h_fmtd32_ = (PimMemTraceData*)malloc(reserved_fmtd_size);
    h_fmtd16_size_ = (size_t*)malloc(sizeof(size_t));
    h_fmtd32_size_ = (size_t*)malloc(sizeof(size_t));

    cl_d_fmtd16_ = clCreateBuffer(context, CL_MEM_READ_WRITE, reserved_fmtd_size, nullptr, &exec_err_);
    clEnqueueFillBuffer(queue, cl_d_fmtd16_, (void*)&zero, sizeof(int), 0, reserved_fmtd_size, 0, NULL, NULL);
    cl_ok(exec_err_);

    cl_d_fmtd16_size_ = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(size_t), nullptr, &exec_err_);
    clEnqueueFillBuffer(queue, cl_d_fmtd16_size_, (void*)&zero, sizeof(int), 0, sizeof(size_t), 0, NULL, NULL);
    cl_ok(exec_err_);

    cl_d_emulator_trace_ = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(PimMemTracer), nullptr, &exec_err_);
    clEnqueueFillBuffer(queue, cl_d_emulator_trace_, (void*)&zero, sizeof(int), 0, sizeof(PimMemTracer), 0, NULL, NULL);
    cl_ok(exec_err_);
#endif
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclPimExecutor::deinitialize(void)
{
    DLOG(INFO) << " [START] " << __FUNCTION__ << " called";
    int ret = 0;
    ret = pim_manager_->free_memory((void*)pim_gemv_tmp_buffer_, MEM_TYPE_PIM);
    ret |= pim_manager_->free_memory((void*)d_srf_bin_buffer_, MEM_TYPE_DEVICE);
    ret |= pim_manager_->free_memory((void*)zero_buffer_, MEM_TYPE_DEVICE);

#ifdef EMULATOR
    clReleaseMemObject(cl_d_fmtd16_);
    clReleaseMemObject(cl_d_fmtd16_size_);
    clReleaseMemObject(cl_d_emulator_trace_);

    free(d_emulator_trace_);
    free(h_fmtd16_);
    free(h_fmtd16_size_);
    free(h_fmtd32_);
    free(h_fmtd32_size_);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

uint8_t* OclPimExecutor::get_crf_bin(PimOpType op_type, int output_size)
{
    uint8_t* crf_bin = pim_crf_generator_->find_crf(op_type, output_size);
    if (crf_bin == nullptr) {
        crf_bin = (uint8_t*)pim_crf_generator_->make_crf_bin(op_type, output_size);
    }
    return crf_bin;
}

#ifdef EMULATOR
void OclPimExecutor::emulator_trace_gen(unsigned int block_size, PimOpType op_type)
{
    h_fmtd16_size_[0] = 0;
    pim_manager_->copy_memory((void*)h_fmtd16_size_, (void*)cl_d_fmtd16_size_, sizeof(size_t), DEVICE_TO_HOST);
    pim_manager_->copy_memory((void*)h_fmtd16_, (void*)cl_d_fmtd16_, sizeof(PimMemTraceData) * max_fmtd_size_,
                              DEVICE_TO_HOST);

    for (size_t i = 1; i < block_size; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(PimMemTraceData));
    }
    h_fmtd16_size_[0] *= block_size;
    pim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, (int*)h_fmtd32_size_, h_fmtd16_, (int)h_fmtd16_size_[0],
                                                     op_type);
}
#endif

int OclPimExecutor::execute_eltwise(PimOpType eltop, PimBo* output, PimBo* operand0, PimBo* operand1, void* stream,
                                    bool block)
{
    DLOG(INFO) << " [START] " << __FUNCTION__ << " called";
    int ret = 0;

    const size_t block_size = 64;
    const size_t local_work_size = 32;
    const size_t global_work_size = block_size * local_work_size;
    int output_size = output->size;
    int align_size = (131072 << 1);
    int num_tile = (output_size + align_size - 1) / align_size;

    uint8_t* crf_bin = get_crf_bin(eltop, output->size);
    int crf_size = CRF_BIN_SIZE;

    cl_ok(clSetKernelArg(eltwise_kernel_, 0, sizeof(cl_mem),
                         (void*)&(((manager::OclBufferObj*)operand0->data)->dev_addr)));
    cl_ok(clSetKernelArg(eltwise_kernel_, 1, sizeof(cl_mem),
                         (void*)&(((manager::OclBufferObj*)operand1->data)->dev_addr)));
    cl_ok(
        clSetKernelArg(eltwise_kernel_, 2, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)output->data)->dev_addr)));
    cl_ok(clSetKernelArg(eltwise_kernel_, 3, sizeof(cl_mem), (void*)&base_address_));
    cl_ok(clSetKernelArg(eltwise_kernel_, 4, sizeof(cl_int), (void*)&num_tile));
    cl_ok(clSetKernelArg(eltwise_kernel_, 5, sizeof(cl_mem), (void*)&crf_bin));
    cl_ok(clSetKernelArg(eltwise_kernel_, 6, sizeof(cl_int), (void*)&crf_size));

#ifdef EMULATOR
    cl_ok(clSetKernelArg(eltwise_kernel_, 7, sizeof(cl_mem), (void*)&cl_d_fmtd16_));
    cl_ok(clSetKernelArg(eltwise_kernel_, 8, sizeof(cl_mem), (void*)&cl_d_fmtd16_size_));
    cl_ok(clSetKernelArg(eltwise_kernel_, 9, sizeof(cl_int), (void*)&fmtd_size_per_ch_));
    cl_ok(clSetKernelArg(eltwise_kernel_, 10, sizeof(cl_mem), (void*)&cl_d_emulator_trace_));

#endif
    exec_err_ =
        clEnqueueNDRangeKernel(queue, eltwise_kernel_, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    clFinish(queue);

#ifdef EMULATOR
    emulator_trace_gen(block_size, OP_ELT_ADD);
    pim_emulator_->execute_elt_op(output, operand0, operand1, h_fmtd32_, (int)h_fmtd32_size_[0], g_pim_base_addr[0]);
#endif
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclPimExecutor::execute_add(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    return execute_eltwise(OP_ELT_ADD, output, operand0, operand1, stream, block);
}

int OclPimExecutor::execute_mul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream, bool block)
{
    return execute_eltwise(OP_ELT_MUL, output, operand0, operand1, stream, block);
}

int OclPimExecutor::execute_relu(PimBo* output, PimBo* pim_data, void* stream, bool block)
{
    const size_t block_size = pbi_->num_pim_chan;
    const size_t local_work_size = 32;
    const size_t global_work_size = block_size * local_work_size;
    int align_size = (131072 << 1);
    int aligned_outsize = ((output->size + align_size - 1) / align_size);
    uint8_t* crf_bin = get_crf_bin(OP_RELU, output->size);
    int crf_size = CRF_BIN_SIZE;

    exec_err_ =
        clSetKernelArg(relu_kernel_, 0, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)pim_data->data)->dev_addr));
    cl_ok(exec_err_);

    exec_err_ =
        clSetKernelArg(relu_kernel_, 1, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)output->data)->dev_addr));
    cl_ok(exec_err_);

    exec_err_ = clSetKernelArg(relu_kernel_, 2, sizeof(cl_mem), (void*)&base_address_);
    cl_ok(exec_err_);

    exec_err_ = clSetKernelArg(relu_kernel_, 3, sizeof(cl_int), (void*)&aligned_outsize);
    cl_ok(exec_err_);

    exec_err_ = clSetKernelArg(relu_kernel_, 4, sizeof(cl_mem), (void*)&crf_bin);
    cl_ok(exec_err_);

    exec_err_ = clSetKernelArg(relu_kernel_, 5, sizeof(cl_int), (void*)&crf_size);
    cl_ok(exec_err_);

#ifdef EMULATOR
    exec_err_ = clSetKernelArg(relu_kernel_, 6, sizeof(cl_mem), (void*)&cl_d_fmtd16_);
    cl_ok(exec_err_);

    exec_err_ = clSetKernelArg(relu_kernel_, 7, sizeof(cl_mem), (void*)&cl_d_fmtd16_size_);
    cl_ok(exec_err_);

    exec_err_ = clSetKernelArg(relu_kernel_, 8, sizeof(cl_int), (void*)&fmtd_size_per_ch_);
    cl_ok(exec_err_);

    exec_err_ = clSetKernelArg(relu_kernel_, 9, sizeof(cl_mem), (void*)&cl_d_emulator_trace_);
    cl_ok(exec_err_);
#endif
    exec_err_ =
        clEnqueueNDRangeKernel(queue, relu_kernel_, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    cl_ok(exec_err_);
    clFinish(queue);

#ifdef EMULATOR
    emulator_trace_gen(block_size, OP_RELU);
    pim_emulator_->execute_relu(output, pim_data, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr[0]);
#endif
    return exec_err_;
}

int OclPimExecutor::execute_copy(PimBo* output, PimBo* pim_data, void* stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;
    const size_t block_size = pbi_->num_pim_chan;
    const size_t local_work_size = 32;
    const size_t global_work_size = block_size * local_work_size;
    size_t output_size = output->size;
    uint8_t* crf_bin = get_crf_bin(OP_COPY, output_size);
    int crf_size = CRF_BIN_SIZE;

    PIM_PROFILE_TICK(RunCopyKernel);
    cl_ok(
        clSetKernelArg(copy_kernel_, 0, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)pim_data->data)->dev_addr)));
    cl_ok(clSetKernelArg(copy_kernel_, 1, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)output->data)->dev_addr)));
    cl_ok(clSetKernelArg(copy_kernel_, 2, sizeof(cl_mem), (void*)&base_address_));
    cl_ok(clSetKernelArg(copy_kernel_, 3, sizeof(cl_int), (void*)&output_size));
    cl_ok(clSetKernelArg(copy_kernel_, 4, sizeof(cl_mem), (void*)&crf_bin));
    cl_ok(clSetKernelArg(copy_kernel_, 5, sizeof(cl_int), (void*)&crf_size));

#ifdef EMULATOR
    cl_ok(clSetKernelArg(copy_kernel_, 6, sizeof(cl_mem), (void*)&cl_d_fmtd16_));
    cl_ok(clSetKernelArg(copy_kernel_, 7, sizeof(cl_mem), (void*)&cl_d_fmtd16_size_));
    cl_ok(clSetKernelArg(copy_kernel_, 8, sizeof(cl_int), (void*)&fmtd_size_per_ch_));
    cl_ok(clSetKernelArg(copy_kernel_, 9, sizeof(cl_mem), (void*)&cl_d_emulator_trace_));
#endif
    cl_ok(clEnqueueNDRangeKernel(queue, copy_kernel_, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    clFinish(queue);
    PIM_PROFILE_TOCK(RunCopyKernel);

#ifdef EMULATOR
    emulator_trace_gen(block_size, OP_COPY);
    pim_emulator_->execute_copy(output, pim_data, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr[0]);
#endif
    return ret;
}

int OclPimExecutor::execute_bn(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                               double epsilon, void* stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    const size_t block_size = pbi_->num_pim_chan;
    const size_t local_work_size = 32;
    const size_t global_work_size = block_size * local_work_size;
    size_t output_size = output->size;
    uint8_t* crf_bin = get_crf_bin(OP_BN, output_size);
    int crf_size = 2 * CRF_BIN_SIZE;
    int align_size = (131072 << 1);
    int num_tile = ((output->size + align_size - 1) / align_size);

    uint8_t* srf_binary = new uint8_t[pbi_->num_pim_chan * pbi_->num_pim_rank * pbi_->trans_size];
    int srf_size = pbi_->num_pim_chan * pbi_->num_pim_rank * pbi_->trans_size;

    pim_crf_generator_->preprocess_srf(beta, gamma, mean, variance, epsilon, srf_binary);
    pim_manager_->copy_memory((void*)d_srf_bin_buffer_, (void*)srf_binary, srf_size, HOST_TO_DEVICE);

    PIM_PROFILE_TICK(RunBNKernel);
    cl_ok(clSetKernelArg(bn_kernel_, 0, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)pim_data->data)->dev_addr)));
    cl_ok(clSetKernelArg(bn_kernel_, 1, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)output->data)->dev_addr)));
    cl_ok(clSetKernelArg(bn_kernel_, 2, sizeof(cl_mem), (void*)&base_address_));
    cl_ok(clSetKernelArg(bn_kernel_, 3, sizeof(cl_int), (void*)&num_tile));
    cl_ok(clSetKernelArg(bn_kernel_, 4, sizeof(cl_mem), (void*)&crf_bin));
    cl_ok(clSetKernelArg(bn_kernel_, 5, sizeof(cl_int), (void*)&crf_size));
    cl_ok(clSetKernelArg(bn_kernel_, 6, sizeof(cl_mem), (void*)&d_srf_bin_buffer_));
    cl_ok(clSetKernelArg(bn_kernel_, 7, sizeof(cl_int), (void*)&srf_size));

#ifdef EMULATOR
    cl_ok(clSetKernelArg(bn_kernel_, 8, sizeof(cl_mem), (void*)&cl_d_fmtd16_));
    cl_ok(clSetKernelArg(bn_kernel_, 9, sizeof(cl_mem), (void*)&cl_d_fmtd16_size_));
    cl_ok(clSetKernelArg(bn_kernel_, 10, sizeof(cl_int), (void*)&fmtd_size_per_ch_));
    cl_ok(clSetKernelArg(bn_kernel_, 11, sizeof(cl_mem), (void*)&cl_d_emulator_trace_));
#endif
    cl_ok(clEnqueueNDRangeKernel(queue, bn_kernel_, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    clFinish(queue);
    PIM_PROFILE_TOCK(RunBNKernel);

#ifdef EMULATOR
    emulator_trace_gen(block_size, OP_BN);
    pim_emulator_->execute_bn(output, pim_data, h_fmtd32_, h_fmtd32_size_[0], g_pim_base_addr[0], nullptr);
#endif
    delete[] srf_binary;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}
int OclPimExecutor::execute_aligned_gemm_tile_accum(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias,
                                                    PimActFunc act_func, void* stream, bool block)
{
    PIM_PROFILE_TICK(PrepareGemmKernel);
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    const size_t block_size = pbi_->num_pim_chan;
    const size_t local_work_size = 64;
    const size_t global_work_size = block_size * local_work_size;

    int n_in_tile = input->bshape.w * sizeof(uint16_t) / pbi_->trans_size / pbi_->num_grf_A;
    int n_out_tile = output->bshape.w / (pbi_->num_pim_chan * pbi_->num_pim_blocks * pbi_->num_grf_B);
    int iter_cnt = weight->bshape.n * weight->bshape.c * weight->bshape.w / PIM_GEMV_OUT_ALIGN;
    int is_bias = (bias != nullptr) ? 1 : 0;
    int is_relu = (act_func == ACT_RELU) ? 1 : 0;

    uint8_t* crf_bin = get_crf_bin(OP_GEMV, input->bshape.w * sizeof(uint16_t));
    int crf_size = CRF_BIN_SIZE;

    cl_kernel gemm_kernel;

    switch (n_in_tile) {
        case 8:
            gemm_kernel = pim_aligned_gemm_bias_relu_8tile_fp16_;
            break;
        default:
            gemm_kernel = pim_aligned_gemm_bias_relu_fp16_;
            break;
    }
    PIM_PROFILE_TOCK(PrepareGemmKernel);

    PIM_PROFILE_TICK(RunGemmKernel);
    cl_ok(clSetKernelArg(gemm_kernel, 0, sizeof(cl_mem), (void*)&base_address_));
    cl_ok(clSetKernelArg(gemm_kernel, 1, sizeof(cl_mem), (void*)&input->data));
    cl_ok(clSetKernelArg(gemm_kernel, 2, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)weight->data)->dev_addr)));
    cl_ok(clSetKernelArg(gemm_kernel, 3, sizeof(cl_mem), (is_bias ? (void*)&bias->data : nullptr)));
    cl_ok(clSetKernelArg(gemm_kernel, 4, sizeof(cl_mem), (void*)&output->data));
    cl_ok(clSetKernelArg(gemm_kernel, 5, sizeof(cl_mem), (void*)&pim_gemv_tmp_buffer_->dev_addr));
    cl_ok(clSetKernelArg(gemm_kernel, 6, sizeof(cl_int), (void*)&iter_cnt));
    cl_ok(clSetKernelArg(gemm_kernel, 7, sizeof(cl_int), (void*)&input->bshape.h));
    cl_ok(clSetKernelArg(gemm_kernel, 8, sizeof(cl_int), (void*)&input->bshape.w));
    cl_ok(clSetKernelArg(gemm_kernel, 9, sizeof(cl_int), (void*)&output->bshape.w));
    cl_ok(clSetKernelArg(gemm_kernel, 10, sizeof(cl_int), (void*)&n_in_tile));
    cl_ok(clSetKernelArg(gemm_kernel, 11, sizeof(cl_int), (void*)&n_out_tile));
    cl_ok(clSetKernelArg(gemm_kernel, 12, sizeof(cl_int), (void*)&is_bias));
    cl_ok(clSetKernelArg(gemm_kernel, 13, sizeof(cl_int), (void*)&is_relu));
    cl_ok(clSetKernelArg(gemm_kernel, 14, sizeof(cl_mem), (void*)&crf_bin));
    cl_ok(clSetKernelArg(gemm_kernel, 15, sizeof(cl_int), (void*)&crf_size));

#ifdef EMULATOR
    cl_ok(clSetKernelArg(gemm_kernel, 16, sizeof(cl_mem), (void*)&cl_d_fmtd16_));
    cl_ok(clSetKernelArg(gemm_kernel, 17, sizeof(cl_mem), (void*)&cl_d_fmtd16_size_));
    cl_ok(clSetKernelArg(gemm_kernel, 18, sizeof(cl_int), (void*)&fmtd_size_per_ch_));
    cl_ok(clSetKernelArg(gemm_kernel, 19, sizeof(cl_mem), (void*)&cl_d_emulator_trace_));
#endif
    exec_err_ = clEnqueueNDRangeKernel(queue, gemm_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    cl_ok(exec_err_);
    clFinish(queue);
    PIM_PROFILE_TOCK(RunGemmKernel);
#ifdef EMULATOR
    PIM_PROFILE_TICK(RunGemmEmulation);
    emulator_trace_gen(block_size, OP_GEMV);
    pim_emulator_->execute_gemm_bias_act(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV, g_pim_base_addr[0],
                                         (uint8_t*)pim_gemv_tmp_buffer_, bias, act_func);

    PIM_PROFILE_TOCK(RunGemmEmulation);
#endif
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int OclPimExecutor::execute_chwise_gemm_tile_accum(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias,
                                                   PimActFunc act_func, void* stream, bool block)
{
    PIM_PROFILE_TICK(PrepareGemmKernel);
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    const size_t block_size = pbi_->num_pim_chan;
    const size_t local_work_size = 64;
    const size_t global_work_size = block_size * local_work_size;

    int n_in_tile = input->bshape.w * sizeof(uint16_t) / pbi_->trans_size / pbi_->num_grf_A;
    int n_out_tile = 1;
    int iter_cnt = (weight->bshape.n * weight->bshape.c) / (PIM_GEMV_OUT_ALIGN / weight->bshape.w);

    int is_bias = (bias != nullptr) ? 1 : 0;
    int is_relu = (act_func == ACT_RELU) ? 1 : 0;

    uint8_t* crf_bin = get_crf_bin(OP_GEMV, input->bshape.w * sizeof(uint16_t));
    int crf_size = CRF_BIN_SIZE;

    cl_kernel gemm_kernel;
    switch (n_in_tile) {
        case 32:
            gemm_kernel = pim_chwise_gemm_bias_relu_32tile_fp16_;
            break;
        default:
            gemm_kernel = pim_chwise_gemm_bias_relu_fp16_;
            break;
    }

    PIM_PROFILE_TOCK(PrepareGemmKernel);

    PIM_PROFILE_TICK(RunGemmKernel);
    cl_ok(clSetKernelArg(gemm_kernel, 0, sizeof(cl_mem), (void*)&base_address_));
    cl_ok(clSetKernelArg(gemm_kernel, 1, sizeof(cl_mem), (void*)&input->data));
    cl_ok(clSetKernelArg(gemm_kernel, 2, sizeof(cl_mem), (void*)&(((manager::OclBufferObj*)weight->data)->dev_addr)));
    cl_ok(clSetKernelArg(gemm_kernel, 3, sizeof(cl_mem), (is_bias ? (void*)&bias->data : nullptr)));
    cl_ok(clSetKernelArg(gemm_kernel, 4, sizeof(cl_mem), (void*)&output->data));
    cl_ok(clSetKernelArg(gemm_kernel, 5, sizeof(cl_mem), (void*)&pim_gemv_tmp_buffer_->dev_addr));
    cl_ok(clSetKernelArg(gemm_kernel, 6, sizeof(cl_int), (void*)&iter_cnt));
    cl_ok(clSetKernelArg(gemm_kernel, 7, sizeof(cl_int), (void*)&input->bshape.h));
    cl_ok(clSetKernelArg(gemm_kernel, 8, sizeof(cl_int), (void*)&input->bshape.w));
    cl_ok(clSetKernelArg(gemm_kernel, 9, sizeof(cl_int), (void*)&output->bshape.w));
    cl_ok(clSetKernelArg(gemm_kernel, 10, sizeof(cl_int), (void*)&n_in_tile));
    cl_ok(clSetKernelArg(gemm_kernel, 11, sizeof(cl_int), (void*)&n_out_tile));
    cl_ok(clSetKernelArg(gemm_kernel, 12, sizeof(cl_int), (void*)&is_bias));
    cl_ok(clSetKernelArg(gemm_kernel, 13, sizeof(cl_int), (void*)&is_relu));
    cl_ok(clSetKernelArg(gemm_kernel, 14, sizeof(cl_mem), (void*)&crf_bin));
    cl_ok(clSetKernelArg(gemm_kernel, 15, sizeof(cl_int), (void*)&crf_size));

#ifdef EMULATOR
    cl_ok(clSetKernelArg(gemm_kernel, 16, sizeof(cl_mem), (void*)&cl_d_fmtd16_));
    cl_ok(clSetKernelArg(gemm_kernel, 17, sizeof(cl_mem), (void*)&cl_d_fmtd16_size_));
    cl_ok(clSetKernelArg(gemm_kernel, 18, sizeof(cl_int), (void*)&fmtd_size_per_ch_));
    cl_ok(clSetKernelArg(gemm_kernel, 19, sizeof(cl_mem), (void*)&cl_d_emulator_trace_));
#endif
    exec_err_ = clEnqueueNDRangeKernel(queue, gemm_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    cl_ok(exec_err_);
    clFinish(queue);

    PIM_PROFILE_TOCK(RunGemmKernel);
#ifdef EMULATOR
    PIM_PROFILE_TICK(RunGemmEmulation);
    emulator_trace_gen(block_size, OP_GEMV);
    pim_emulator_->execute_gemm_bias_act(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV, g_pim_base_addr[0],
                                         (uint8_t*)pim_gemv_tmp_buffer_, bias, act_func);

    PIM_PROFILE_TOCK(RunGemmEmulation);
#endif
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}
int OclPimExecutor::execute_ocl_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                                     void* stream, bool block)
{
    int ret = 0;
    if (weight->data_layout_type == PimDataLayoutType::CHWISE_GEMM_WEIGHT)
        ret = execute_chwise_gemm_tile_accum(output, input, weight, bias, act_func, stream, block);
    else if (weight->data_layout_type == PimDataLayoutType::ALIGNED_GEMM_WEIGHT)
        ret = execute_aligned_gemm_tile_accum(output, input, weight, bias, act_func, stream, block);
    else
        DLOG(ERROR) << "Provided layout is not supported in GEMM call";

    return ret;
}
int OclPimExecutor::execute_gemv(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                                 void* stream, bool block)
{
    int ret = 0;
    DLOG(ERROR) << "OCL Gemv GPU UnImplemented ";
    return ret;
}
int OclPimExecutor::execute_custom_gemv(PimBo* output, PimBo* operand0, PimBo* operand1, bool is_gemv_add, void* stream,
                                        bool block)
{
    int ret = 0;
    DLOG(ERROR) << "OCL Custom Gemv GPU UnImplemented ";
    return ret;
}
int OclPimExecutor::execute_custom_gemv_add(PimBo* output, PimBo* operand0, PimBo* operand1, PimBo* operand2, bool relu,
                                            void* stream, bool block)
{
    int ret = 0;
    DLOG(ERROR) << "OCL  Custom Gemv ADD GPU UnImplemented ";
    return ret;
}
int OclPimExecutor::execute_gemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func,
                                 void* stream, bool block)
{
    int ret = 0;
    if (kernel_type_ == CUSTOM_GPU) {
        // ret = this->execute_gemv(output, input, weight, bias, act_func, stream, block);
        DLOG(ERROR) << "OCL Custom GPU UnImplemented ";
    } else if (kernel_type_ == PIM) {
        PimBo* pim_wei;
        if (weight->data_layout_type == PimDataLayoutType::RAW) {
            pim_wei = pim_runtime_->get_preloaded_pim_gemm_weight(weight, gemm_order_);
        } else {
            // Assume that user has provided correct layout
            pim_wei = weight;
        }
        ret = this->execute_ocl_gemm(output, input, pim_wei, bias, act_func, stream, block);
    } else {
        if (is_pim_applicable(weight, gemm_order_)) {
            PimBo* pim_wei;
            if (weight->data_layout_type == PimDataLayoutType::RAW) {
                pim_wei = pim_runtime_->get_preloaded_pim_gemm_weight(weight, gemm_order_);
            } else {
                // Assume that user has provided correct layout
                pim_wei = weight;
            }
            ret = this->execute_ocl_gemm(output, input, pim_wei, bias, act_func, stream, block);
        } else {
            std::cout << "Pim Not Applicable" << std::endl;
            ret = this->execute_gemv(output, input, weight, bias, act_func, stream, block);
        }
    }

    return ret;
}

int OclPimExecutor::execute_sync(void* stream)
{
    cl_ok(clFinish(queue));
    return 0;
}
}  // namespace executor
}  // namespace runtime
}  // namespace pim
