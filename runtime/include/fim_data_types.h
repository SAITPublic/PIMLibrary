#ifndef _FIM_DATA_TYPE_H_
#define _FIM_DATA_TYPE_H_

#include <stddef.h>

#define __FIM_API__

#define FP16 _Float16
#define INT8 char

inline float convertH2F(_Float16 h_val)
{
    float f_val = h_val;
    return f_val;
}

inline FP16 convertF2H(float f_val)
{
    FP16 h_val(f_val);
    return h_val;
}

typedef enum __FimRuntimeType {
    RT_TYPE_HIP,
    RT_TYPE_OPENCL,
} FimRuntimeType;

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

#endif /* _FIM_DATA_TYPE_H_ */
