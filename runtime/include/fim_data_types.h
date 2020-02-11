#ifndef _FIM_DATA_TYPE_H_
#define _FIM_DATA_TYPE_H_

#include <stddef.h>
#include "hip/hip_fp16.h"

#define __FIM_API__

#define FP16 _Float16
#define INT8 char

inline float convertH2F(FP16 h_val) { return __half2float(h_val); }

inline FP16 convertF2H(float f_val) { return __float2half(f_val); }

typedef enum __FimRuntimeType {
    RT_TYPE_HIP,
    RT_TYPE_OPENCL,
} FimRuntimeType;

typedef enum __FimAddrMap {
    AMDGPU_VEGA20,
} FimAddrMap;

typedef enum __FimMemType {
    MEM_TYPE_HOST,
    MEM_TYPE_DEVICE,
    MEM_TYPE_FIM,
} FimMemType;

typedef enum __FimMemcpyType {
    HOST_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_FIM,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    DEVICE_TO_FIM,
    FIM_TO_HOST,
    FIM_TO_DEVICE,
    FIM_TO_FIM,
} FimMemcpyType;

typedef enum __FimOpType {
    OP_GEMV,
    OP_ELT_ADD,
    OP_ELT_MUL,
    OP_BATCH_NORM,
    OP_RELU,
    OP_DUMMY,
} FimOpType;

typedef enum __FimPrecision {
    FIM_FP16,
    FIM_INT8,
} FimPrecision;

typedef struct __FimBufferObject {
    size_t size;
    FimMemType memType;
    void* data;
} FimBo;

typedef enum __FimBankType {
    EVEN_BANK,
    ODD_BANK,
    ALL_BANK,
} FimBankType;

typedef enum __FimMode {
    SB_MODE,
    HAB_MODE,
    HAB_FIM,
};

#endif /* _FIM_DATA_TYPE_H_ */
