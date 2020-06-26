#ifndef _FIM_RUNTIME_API_H_
#define _FIM_RUNTIME_API_H_

#include "fim_data_types.h"

/** @mainpage FIM SDK
 *
 * @section intro_sec Documentation of FIM SDK
 *
 * This documentation details about FIM SDK usage
 * @ref FIM-API-Documentation "FIM SDK API Documentation"
 *
 */

/** @file fim_runtime_api.h
 *   @brief FIM API Documentation
 */

/**
 * @defgroup FIM-API-Documentation "FIM API Documentation"
 * @{
 */

/**
 * @brief Initialization API.
 *
 * This API initializes FIM System with either OpenCL/HIP runtime
 * Precision supported are FP16 and INT8
 *
 * @param rt_type       SDK runtime options ( RT_TYPE_HIP, RT_TYPE_OPENCL)
 * @param FimPrecision  Options to choose FIM operations precision ( FIM_FP16, FIM_INT8)
 *
 * @return Return success/failure
 */
__FIM_API__ int FimInitialize(FimRuntimeType rt_type = RT_TYPE_HIP, FimPrecision = FIM_FP16);

/**
 * @brief Deinitialize FIM
 *
 * This call need to be called when FIM need to be reset and clear resourses.
 * @return Return success/failure
 */
__FIM_API__ int FimDeinitialize(void);

/**
 * @brief Creates FIM buffer object of size w,h,c,n with precision (INT8/FIM16)
 *
 * @param w width of buffer object
 * @param h height of buffer object
 * @param c number of channels in buffer object
 * @param n number of batches in buffer object
 * @param precision precision of buffer object (INT8/ FIM16)
 * @param mem_type  type of memory ( FIM/GPU/HOST)
 *
 * @return Pointer to buffer object.
 */
__FIM_API__ FimBo* FimCreateBo(int w, int h, int c, int n, FimPrecision precision, FimMemType mem_type);

/**
 * @brief Create FIM buffer object with fim descriptor
 *
 * @param fim_desc FIM descriptor
 * @param mem_type type of memory need to be allocated (FIM/GPU/HOST)
 * @param mem_flag Describes operation for which buffer is used for( element wise or gemv)
 *
 * @return Pointer to buffer object
 */
__FIM_API__ FimBo* FimCreateBo(FimDesc* fim_desc, FimMemType mem_type, FimMemFlag mem_flag = ELT_OP);

/**
 * @brief Destroy Buffer object
 *
 * @param fim_bo Buffer object pointer to be destroyed.
 *
 * @return
 */
__FIM_API__ int FimDestroyBo(FimBo* fim_bo);

/**
 * @brief Create FIM descriptor using parameters passed
 *
 * @param n number of batches
 * @param c number of channels
 * @param h height of buffer
 * @param w width of buffer
 * @param precision precision of buffer
 *
 * @return Return success/failure
 */
__FIM_API__ FimDesc* FimCreateDesc(int n, int c, int h, int w, FimPrecision precision);

/**
 * @brief Destroy FIM descriptor
 *
 * @param fim_desc pointer to descriptor to be destroyed
 *
 * @return Return success/failure
 */
__FIM_API__ int FimDestroyDesc(FimDesc* fim_desc);

/**
 * @brief Alloc Memory of size and type mem_type
 *
 * This API provices allocation on GPU/FIM or HOST memory of required size
 *
 * @param ptr [out] pointer to allocated memory
 * @param size size of memory need to be allocated
 * @param mem_type Type of memory need to be allocated [ GPU, FIM, HOST]
 *
 * @return success/failure
 */
__FIM_API__ int FimAllocMemory(void** ptr, size_t size, FimMemType mem_type);

/**
 * @brief Alloc memory using buffer object descriptor
 *
 * @param fim_bo Buffer object.
 *
 * @return success/failure
 */
__FIM_API__ int FimAllocMemory(FimBo* fim_bo);

/**
 * @brief Free Allocated memory from FIM API
 *
 * @param ptr pointer to memory need to be freed
 * @param mem_type type of memory
 *
 * @return success/failure
 */
__FIM_API__ int FimFreeMemory(void* ptr, FimMemType mem_type);

/**
 * @brief Free allocated memory from FIM API
 *
 * @param fim_bo buffer object to be destroyed
 *
 * @return successs/failure
 */
__FIM_API__ int FimFreeMemory(FimBo* fim_bo);

/**
 * @brief Convert Data layout of operations  type op_type
 *
 * This API converts data layout from App format to FIM data layout
 * Uses Buffer pointers for access
 *
 * @param dst destination address of converted data
 * @param src source address of data to be converted
 * @param size size of buffer
 * @param op_type operation type ( elt add, mul, gemv)
 *
 * @return success/failure
 */
__FIM_API__ int FimConvertDataLayout(void* dst, void* src, size_t size, FimOpType op_type);

/**
 * @brief Convert data layout of operations type op_type
 *
 * This API converts data layout from App format to FIM. uses Fim buffer objects
 *
 * @param dst destination buffer object
 * @param src source buffer object
 * @param op_type type of operation ( elt add, mul, gemv)
 *
 * @return
 */
__FIM_API__ int FimConvertDataLayout(FimBo* dst, FimBo* src, FimOpType op_type);

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
__FIM_API__ int FimConvertDataLayout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type);

/**
 * @brief Copies data from source to destination
 *
 * @param dst destination address of buffer
 * @param src source address of buffer
 * @param size size of buffer to be copied
 * @param cpy_type type of memory transfer ( HOST to GPU, GPU to HOST, GPU to FIM etc)
 *
 * @return
 */
__FIM_API__ int FimCopyMemory(void* dst, void* src, size_t size, FimMemCpyType cpy_type);

/**
 * @brief Copies data from source buffer object o destination buffer object
 *
 * @param dst destination buffer object
 * @param src source buffer object
 * @param cpy_type type of memory transfer ( HOST to GPU, GPU to HOST, GPU to FIM etc)
 *
 * @return
 */
__FIM_API__ int FimCopyMemory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type);

/**
 * @brief Execute Add vector operation on FIM
 *
 * Executes add operations using FIM buffer objects
 *
 * @param output output Buffer object
 * @param operand0 input 1 of add operations- conveted data
 * @param operand1 input 2 of add operations
 *
 * @return success/failure
 */
__FIM_API__ int FimExecuteAdd(FimBo* output, FimBo* operand0, FimBo* operand1);

/**
 * @brief Execute add scalar operation on FIM
 *
 * Executes add scalar operation using FIM
 *
 * @param output output buffer object
 * @param scalar scalar value to be added
 * @param vector input vector for add operations
 *
 * @return success/failure
 */
__FIM_API__ int FimExecuteAdd(FimBo* output, void* scalar, FimBo* vector);

/**
 * @brief Executes Mul vector operation in FIM
 *
 * @param output output buffer object
 * @param operand0 first operand for Mul operations ( converted data)
 * @param operand1 second operand for Mul Operations.
 *
 * @return
 */
__FIM_API__ int FimExecuteMul(FimBo* output, FimBo* operand0, FimBo* operand1);

/**
 * @brief Executes Mul Scalar operation in FIM
 *
 * @param output output buffer object
 * @param scalar scalar value to be multiplied to vector
 * @param vector vector input
 *
 * @return success/failure
 */
__FIM_API__ int FimExecuteMul(FimBo* output, void* scalar, FimBo* vector);

/**
 * @brief Executes FIM Relu operations
 *
 * @param output Output Buffer object
 * @param fim_data input buffer object
 *
 * @return success/failure
 */
__FIM_API__ int FimExecuteRelu(FimBo* output, FimBo* fim_data);

/**
 * @brief Executes FIM GEMV operation
 *
 * This API provides interface for FIM GEMV operations.
 * For FIM GemV operations, weights(kernel values) need to be preprocessed with Convert Data FIM API
 * Ouptu values are placed in FIM area and need to be transfered to GPU or HOST memory as per requirements
 *
 * @param output output buffer object of gemv
 * @param operand0 input operand 0 ( weights). Should be of FIM Area
 * @param operand1 vector input
 *
 * @return success or failure
 */
__FIM_API__ int FimExecuteGEMV(FimBo* output, FimBo* operand0, FimBo* operand1);

/**
 * @brief Executes Batch normalization operation.
 *
 * @param output output buffer object for BN operation
 * @param fim_data input buffer object ( Should be of FIM Area)
 * @param beta Fim Buffer object having beta values for BN operation
 * @param gamma Fim Buffer object having gamma values for BN operation
 * @param mean Fim Buffer object having mean values for BN operation
 * @param variance Fim Buffer object having variance for BN operation
 * @param epsilon epsilon value for BN operation
 *
 * @return success/failure
 */
__FIM_API__ int FimExecuteBN(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance,
                             double epsilon);

/**
 * @brief Execute Dummy operation in FIM
 *
 * @return
 */
__FIM_API__ int FimExecuteDummy(void);

/**@}*/
#endif /* _FIM_RUNTIME_API_H_ */
