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
 * @param rt_type       SDK runtime options ( RT_TYPE_HIP, RT_TYPE_OPENCL)
 * @param PimPrecision  Options to choose PIM operations precision ( PIM_FP16, PIM_INT8)
 *
 * @return Return success/failure
 */
__PIM_API__ int PimInitialize(PimRuntimeType rt_type = RT_TYPE_HIP, PimPrecision = PIM_FP16);

/**
 * @brief Deinitialize PIM
 *
 * This call need to be called when PIM need to be reset and clear resourses.
 * @return Return success/failure
 */
__PIM_API__ int PimDeinitialize(void);

/**
 * @brief Creates PIM buffer object of size w,h,c,n with precision (INT8/PIM16)
 *
 * @param w width of buffer object
 * @param h height of buffer object
 * @param c number of channels in buffer object
 * @param n number of batches in buffer object
 * @param precision precision of buffer object (INT8/ PIM16)
 * @param mem_type  type of memory ( PIM/GPU/HOST)
 *
 * @return Pointer to buffer object.
 */
__PIM_API__ PimBo* PimCreateBo(int w, int h, int c, int n, PimPrecision precision, PimMemType mem_type);

/**
 * @brief Create PIM buffer object with pim descriptor
 *
 * @param pim_desc PIM descriptor
 * @param mem_type type of memory need to be allocated (PIM/GPU/HOST)
 * @param mem_flag Describes operation for which buffer is used for( element wise or gemv)
 *
 * @return Pointer to buffer object
 */
__PIM_API__ PimBo* PimCreateBo(PimDesc* pim_desc, PimMemType mem_type, PimMemFlag mem_flag = ELT_OP);

/**
 * @brief Destroy Buffer object
 *
 * @param pim_bo Buffer object pointer to be destroyed.
 *
 * @return
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
 * @return Return success/failure
 */
__PIM_API__ PimDesc* PimCreateDesc(int n, int c, int h, int w, PimPrecision precision, PimOpType op_type = OP_ELT_ADD);

/**
 * @brief Destroy PIM descriptor
 *
 * @param pim_desc pointer to descriptor to be destroyed
 *
 * @return Return success/failure
 */
__PIM_API__ int PimDestroyDesc(PimDesc* pim_desc);

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
 * @brief Convert Data layout of operations  type op_type
 *
 * This API converts data layout from App format to PIM data layout
 * Uses Buffer pointers for access
 *
 * @param dst destination address of converted data
 * @param src source address of data to be converted
 * @param size size of buffer
 * @param op_type operation type ( elt add, mul, gemv)
 *
 * @return success/failure
 */
__PIM_API__ int PimConvertDataLayout(void* dst, void* src, size_t size, PimOpType op_type);

/**
 * @brief Convert data layout of operations type op_type
 *
 * This API converts data layout from App format to PIM. uses Pim buffer objects
 *
 * @param dst destination buffer object
 * @param src source buffer object
 * @param op_type type of operation ( elt add, mul, gemv)
 *
 * @return
 */
__PIM_API__ int PimConvertDataLayout(PimBo* dst, PimBo* src, PimOpType op_type);

/**
 * @brief Convert data layout  of operations type op_type
 *
 * @param dst destination address of converted data
 * @param src0 source address 1
 * @param src1 source address 2
 * @param op_type operation type
 *
 * @return success/failure
 */
__PIM_API__ int PimConvertDataLayout(PimBo* dst, PimBo* src0, PimBo* src1, PimOpType op_type);

/**
 * @brief Copies data from source to destination
 *
 * @param dst destination address of buffer
 * @param src source address of buffer
 * @param size size of buffer to be copied
 * @param cpy_type type of memory transfer ( HOST to GPU, GPU to HOST, GPU to PIM etc)
 *
 * @return
 */
__PIM_API__ int PimCopyMemory(void* dst, void* src, size_t size, PimMemCpyType cpy_type);

/**
 * @brief Copies data from source buffer object o destination buffer object
 *
 * @param dst destination buffer object
 * @param src source buffer object
 * @param cpy_type type of memory transfer ( HOST to GPU, GPU to HOST, GPU to PIM etc)
 *
 * @return
 */
__PIM_API__ int PimCopyMemory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type);

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
 * @return
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
 * @brief Executes PIM GEMV operation
 *
 * This API provides interface for PIM GEMV operations.
 * For PIM GemV operations, weights(kernel values) need to be preprocessed with Convert Data PIM API
 * Ouptu values are placed in PIM area and need to be transfered to GPU or HOST memory as per requirements
 *
 * @param output output buffer object of gemv
 * @param operand0 input operand 0 ( weights). Should be of PIM Area
 * @param operand1 vector input
 * @param stream void pointer to stream identifier. default=nullptr
 * @param block enable/disable synchronization. default=false
 *
 * @return success or failure
 */

__PIM_API__ int PimExecuteGemv(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream = nullptr,
                               bool block = false);

/**
 * @brief Executes PIM GEMV + Add operation
 *
 * This API provides interface for PIM GEMV + Add operation.
 * It performs output = output + pim_gemv_result
 *
 * @param output output buffer object of gemv
 * @param operand0 input operand 0 ( weights). Should be of PIM Area
 * @param operand1 vector input
 *
 * @return success or failure
 */
__PIM_API__ int PimExecuteGemvAdd(PimBo* output, PimBo* operand0, PimBo* operand1, void* stream = nullptr,
                                  bool block = false);

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
 * @return
 */
__PIM_API__ int PimSynchronize(void* stream = nullptr);

/**
 * @brief Execute Dummy operation in PIM
 *
 * @return
 */
__PIM_API__ int PimExecuteDummy(void);

/**
 * @brief Create Bundle used for gemv
 *
 * Contains input and weight addresses that are used several times for gemv operation
 *
 * @param input input buffer object for gemv
 * @param weight weight buffer object for gemv
 * @param output output buffer object for gemv
 *
 * @return Pointer to created gemv-bundle
 */
__PIM_API__ PimGemvBundle* PimCreateGemvBundle(PimBo* input, PimBo* weight, PimBo* output);

/**
 * @brief Find proper gemv-bundle to use
 *
 * Finds a gemv-bundle that has weight and input memory slot to do gemv.
 *
 * @param w_addr weight address is searching key to find gemv-bundle
 *
 * @return Pointer to found gemv-bundle
 */
__PIM_API__ PimGemvBundle* PimFindGemvBundle(uint64_t w_addr);

/**
 * @brief Insert gemv-bundle to the map
 *
 * Insert gemv-bundle if there is no exsisting weight related to the given weight address.
 *
 * @param w_addr weight address is searching key to find gemv-bundle
 * @param pim_addr pointer to gemv-bundle to be added on the map
 *
 * @return success or failure
 */
__PIM_API__ int PimInsertGemvBundle(uint64_t w_addr, PimGemvBundle* bundle);

/**@}*/
#endif /* _PIM_RUNTIME_API_H_ */
