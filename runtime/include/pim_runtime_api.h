/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_RUNTIME_API_H_
#define _PIM_RUNTIME_API_H_

#include <api/pim_compiler.h>
#include "pim_data_types.h"

/** @mainpage PIM SDK
 *
 * @section intro_sec Documentation of PIM SDK
 *
 * This documentation details about PIM SDK usage
 * @ref PIM-API-Documentation "PIM SDK API Documentation"
 *
 */

/** @file pim_runtime_api.h
 *   @brief PIM API Documentation
 */

/**
 * @defgroup PIM-API-Documentation "PIM API Documentation"
 * @{
 */

/**
 * @brief Initialization API.
 *
 * This API initializes PIM System with either OpenCL/HIP runtime
 * Precision supported are FP16 and INT8
 *
 * @param rt_type       SDK runtime options (RT_TYPE_HIP, RT_TYPE_OPENCL)
 * @param PimPrecision  Options to choose PIM operations precision (PIM_FP16, PIM_INT8)
 *
 * @return Return success/failure
 */
__PIM_API__ int PimInitialize(PimRuntimeType rt_type = RT_TYPE_HIP, PimPrecision = PIM_FP16);

/**
 * @brief Deinitialize PIM
 *
 * This call need to be called when PIM need to be reset and clear resourses.
 *
 * @return Return success/failure
 */
__PIM_API__ int PimDeinitialize(void);

/**
 * @brief Set PimDevice for Execution
 *
 * This call set the current device for execution to device id
 *
 * @param device_id device id value to set
 *
 * @return Return success/failure
 */
__PIM_API__ int PimSetDevice(uint32_t device_id);

/**
 * @brief Get PimDevice to check device id
 *
 * This call get the current device to check current working device id
 *
 * @param device_id device id variable to get
 *
 * @return Return success/failure
 */
__PIM_API__ int PimGetDevice(uint32_t* device_id);

/**
 * @brief Creates PIM buffer object of size n,c,h,w with precision (PIM_INT8, PIM_FP16)
 *
 * @param n number of batches in buffer object
 * @param c number of channels in buffer object
 * @param h height of buffer object
 * @param w width of buffer object
 * @param precision precision of buffer object (PIM_INT8, PIM_FP16)
 * @param mem_type  type of memory (PIM/GPU/HOST)
 * @param user_ptr pointer indicating user pre-allocated buffer (optional)
 * @param transposed representing whether the content has been transposed or not (optional)
 *
 * @return Pointer to buffer object.
 */
__PIM_API__ PimBo* PimCreateBo(int n, int c, int h, int w, PimPrecision precision, PimMemType mem_type,
                               void* user_ptr = nullptr, bool transposed = false);

/**
 * @brief create pim buffer object with pim descriptor
 *
 * @param pim_desc pim descriptor
 * @param mem_type type of memory need to be allocated (pim/gpu/host)
 * @param mem_flag describes operation for which buffer is used for (element wise or gemm)
 * @param user_ptr external memory passed by user. if passed, bo is created with user pointer.
 *                 if nullptr, pim library does the allocation
 *
 * @return pointer to buffer object
 */
__PIM_API__ PimBo* PimCreateBo(PimDesc* pim_desc, PimMemType mem_type, PimMemFlag mem_flag = ELT_OP,
                               void* user_ptr = nullptr);

/**
 * @brief create pim buffer object with pim gemm descriptor
 *
 * @param pim_gemm_desc pim gemm descriptor
 * @param mem_type type of memory need to be allocated (pim/gpu/host)
 * @param mem_flag describes operation for which buffer is used for (element wise or gemm)
 * @param user_ptr external memory passed by user. if passed, bo is created with user pointer.
 *                 if nullptr, pim library does the allocation
 *
 * @return pointer to buffer object
 */
__PIM_API__ PimBo* PimCreateBo(PimGemmDesc* pim_gemm_desc, PimMemType mem_type, PimMemFlag mem_flag,
                               void* user_ptr = nullptr);

/**
 * @brief Destroy Buffer object
 *
 * @param pim_bo Buffer object pointer to be destroyed.
 *
 * @return successs/failure
 */
__PIM_API__ int PimDestroyBo(PimBo* pim_bo);

/**
 * @brief Create PIM descriptor using parameters passed
 *
 * @param n number of batches
 * @param c number of channels
 * @param h height of buffer
 * @param w width of buffer
 * @param precision precision of buffer
 *
 * @return PimDesc structure
 */
__PIM_API__ PimDesc* PimCreateDesc(int n, int c, int h, int w, PimPrecision precision, PimOpType op_type = OP_ELT_ADD);

/**
 * @brief Destroy PIM descriptor
 *
 * @param pim_desc pointer to descriptor to be destroyed
 *
 * @return success/failure
 */
__PIM_API__ int PimDestroyDesc(PimDesc* pim_desc);

/**
 * @brief Create PIM GEMM descriptor using parameters passed
 *
 * This API makes dimension for input(nchw) x weight(nchw) = output(nchw)
 *
 * @param n number of batches
 * @param c number of channels
 * @param inout_h height of input and output buffer
 * @param in_w width of input buffer and height of weight buffer
 * @param out_w width of weight and output buffer
 * @param precision precision of buffer
 * @param transposed representing whether the content has been transposed or not (optional)
 *
 * @return PimGemmDesc structure
 */
__PIM_API__ PimGemmDesc* PimCreateGemmDesc(int n, int c, int inout_h, int in_w, int out_w, PimPrecision precision,
                                           bool transposed = false);

/**
 * @brief Destroy PIM GEMM descriptor
 *
 * @param pim_gemm_desc pointer to descriptor to be destroyed
 *
 * @return success/failure
 */
__PIM_API__ int PimDestroyGemmDesc(PimGemmDesc* pim_gemm_desc);

/**
 * @brief Alloc Memory of size and type mem_type
 *
 * This API provices allocation on GPU/PIM or HOST memory of required size
 *
 * @param ptr [out] pointer to allocated memory
 * @param size size of memory need to be allocated
 * @param mem_type Type of memory need to be allocated [ GPU, PIM, HOST]
 *
 * @return success/failure
 */
__PIM_API__ int PimAllocMemory(void** ptr, size_t size, PimMemType mem_type);

/**
 * @brief Alloc memory using buffer object descriptor
 *
 * @param pim_bo Buffer object.
 *
 * @return success/failure
 */
__PIM_API__ int PimAllocMemory(PimBo* pim_bo);

/**
 * @brief Free Allocated memory from PIM API
 *
 * @param ptr pointer to memory need to be freed
 * @param mem_type type of memory
 *
 * @return success/failure
 */
__PIM_API__ int PimFreeMemory(void* ptr, PimMemType mem_type);

/**
 * @brief Free allocated memory from PIM API
 *
 * @param pim_bo buffer object to be destroyed
 *
 * @return successs/failure
 */
__PIM_API__ int PimFreeMemory(PimBo* pim_bo);

/**
 * @brief Copies data from source to destination
 *
 * @param dst destination address of buffer
 * @param src source address of buffer
 * @param size size of buffer to be copied
 * @param cpy_type type of memory transfer (HOST to GPU, GPU to HOST, GPU to PIM etc)
 *
 * @return successs/failure
 */
__PIM_API__ int PimCopyMemory(void* dst, void* src, size_t size, PimMemCpyType cpy_type);

/**
 * @brief Copies data from source buffer object o destination buffer object
 *
 * @param dst destination buffer object
 * @param src source buffer object
 * @param cpy_type type of memory transfer (HOST to GPU, GPU to HOST, GPU to PIM etc)
 *
 * @return successs/failure
 */
__PIM_API__ int PimCopyMemory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type);

/**
 * @brief Copies a rectangular 3D slice between source and destination.
 *
 * @param copy_params 3D memory copy parameters for the rectangular copy.
 *
 * @return success/failure
 */
__PIM_API__ int PimCopyMemoryRect(const PimCopy3D* copy_params);

/**
 * @brief Creates a new stream/CommandQueue based on runtime type
 *
 * return a void* to a stream object of current runtime type
 *
 * @param rt_type runtime type
 *
 * @return void* stream object
 */
__PIM_API__ void* createStream(PimRuntimeType rt_type);

/**
 * @brief Execute Add vector operation on PIM
 *
 * Executes add operations using PIM buffer objects
 *
 * @param output output Buffer object
 * @param operand0 input 1 of add operations- conveted data
 * @param operand1 input 2 of add operations
 * @param stream void pointer to stream identifier. default=nullptr
 * @param block enable/disable synchronization. default=false
 *
 * @return success/failure
 */
__PIM_API__ int PimExecuteAdd(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream = nullptr,
                              bool block = false);

/**
 * @brief Execute add scalar operation on PIM
 *
 * Executes add scalar operation using PIM
 *
 * @param output output buffer object
 * @param scalar scalar value to be added
 * @param vector input vector for add operations
 *
 * @return success/failure
 */
__PIM_API__ int PimExecuteAdd(PimBo* output, void* scalar, PimBo* vector, void* stream = nullptr, bool block = false);

/**
 * @brief Executes Mul vector operation in PIM
 *
 * @param output output buffer object
 * @param operand0 first operand for Mul operations ( converted data)
 * @param operand1 second operand for Mul Operations.
 * @param stream void pointer to stream identifier. default=nullptr
 * @param block enable/disable synchronization. default=false
 *
 * @return success/failure
 */
__PIM_API__ int PimExecuteMul(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream = nullptr,
                              bool block = false);

/**
 * @brief Executes Mul Scalar operation in PIM
 *
 * @param output output buffer object
 * @param scalar scalar value to be multiplied to vector
 * @param vector vector input
 * @param stream void pointer to stream identifier. default=nullptr
 * @param block enable/disable synchronization. default=false
 *
 * @return success/failure
 */
__PIM_API__ int PimExecuteMul(PimBo* output, void* scalar, PimBo* vector, void* stream = nullptr, bool block = false);

/**
 * @brief Executes PIM Relu operations
 *
 * @param output Output Buffer object
 * @param pim_data input buffer object
 * @param stream void pointer to stream identifier. default=nullptr
 * @param block enable/disable synchronization. default=false
 *
 * @return success/failure
 */
__PIM_API__ int PimExecuteRelu(PimBo* output, PimBo* pim_data, void* stream = nullptr, bool block = false);

/**
 * @brief Executes PIM GEMM operation
 *
 * This API provides interface for PIM GEMM operations.
 * For PIM GemM operations, Input and Weight should be placed in GPU memory area.
 * and weight(kernel values) are internally replaced in PIM area.
 * Output values are also placed in GPU area.
 *
 * @param output output buffer object
 * @param input input buffer object
 * @param weight weight buffer object
 * @param bias bias buffer object
 * @param act_func activation function for PIM GEMM output
 * @param stream void pointer to stream identifier. default=nullptr
 * @param block enable/disable synchronization. default=false
 *
 * @return success/failure
 */
__PIM_API__ int PimExecuteGemm(PimBo* output, PimBo* input, PimBo* weight, PimBo* bias, PimActFunc act_func = NONE,
                               void* stream = nullptr, bool block = false);

/**
 * @brief Executes Batch normalization operation.
 *
 * @param output output buffer object for BN operation
 * @param pim_data input buffer object ( Should be of PIM Area)
 * @param beta Pim Buffer object having beta values for BN operation
 * @param gamma Pim Buffer object having gamma values for BN operation
 * @param mean Pim Buffer object having mean values for BN operation
 * @param variance Pim Buffer object having variance for BN operation
 * @param epsilon epsilon value for BN operation
 * @param stream void pointer to stream identifier. default=nullptr
 * @param block enable/disable synchronization. default=false
 *
 * @return success/failure
 */
__PIM_API__ int PimExecuteBN(PimBo* output, PimBo* pim_data, PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance,
                             double epsilon, void* stream = nullptr, bool block = false);

/**
 * @brief Synchronization call for PIM commands
 *
 * This API blocks execution until all previously issues commands are completed
 *
 * @return success or failure
 */
__PIM_API__ int PimSynchronize(void* stream = nullptr);

/**
 * @brief Execute Dummy operation in PIM
 *
 * @return success/failure
 */
__PIM_API__ int PimExecuteDummy(void);

/**
 * @brief Returns source buffer reordered to desired type
 *
 * @param src source address of buffer
 * @param result_type desired layout type for source buffer
 * @param save_for_reuse enable/disable caching.
 *                       preferable for multiple usage of dst buffer.
 *                       default=false
 *
 * @return reordered buffer
 */
__PIM_API__ PimBo* PimConvertGemmWeight(PimBo* src, bool save_for_reuse = false);

#if PIM_COMPILER_ENABLE == 1
/**
 * @brief Create PIM Target
 *
 * This call create the target with Runtime type (HIP/OpenCL), precision and device (GPU)
 *
 * @param Runtime type (HIP/OpenCL)
 * @param PimPrecision FP16
 * @param PimDevice (GPU)
 *
 * @return Return Target object
 */
__PIM_API__ PimTarget* PimCreateTarget(PimRuntimeType rt_type, PimPrecision precision, PimDevice device);

/**
 * @brief Destroy Target object
 *
 * @param PimTarget object pointer to be destroyed.
 *
 * @return successs/failure
 */
__PIM_API__ int PimDestroyTarget(PimTarget* target);

/**
 * @brief Build PIM Program, compile code
 *
 * This call builds PIM program using PIM Compiler to get compiled code gpu kernel, CRF Binary
 *
 * @param Output Var
 * @param vector of input buffers
 * @param vector of input PIM Buffer Objects
 * @param PimTarget object
 * @param compile options
 *
 * @return Return PIM Compiled Object
 */
__PIM_API__ PimCompiledObj* PimBuildProgram(pimc::frontend::Var output, std::vector<pimc::frontend::Buffer> inputs,
                                            std::vector<PimBo*> input_pimbo, PimTarget* target,
                                            std::string compile_opts = "-O0");

/**
 * @brief Execute PIM Program
 *
 * This call executes the compiled code, launches the target kernel
 *
 * @param PIM Compiled object- CRF Binary, kernel
 * @param PimTarget object
 * @param launch options
 *
 * @return Return output PIM Buffer Object
 */
__PIM_API__ PimBo* PimExecuteProgram(PimCompiledObj* obj, PimTarget* target, std::string launch_opts = "");

/**
 * @brief Destroy Program
 *
 * @param PimCompiledObj object pointer to be destroyed.
 *
 * @return successs/failure
 */
__PIM_API__ int PimDestroyProgram(PimCompiledObj* obj);
#endif
/**@}*/

#endif /* _PIM_RUNTIME_API_H_ */
