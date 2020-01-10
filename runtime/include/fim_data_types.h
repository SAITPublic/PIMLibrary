#ifndef FIM_DATA_TYPE_H_
#define FIM_DATA_TYPE_H_

#define __FIM_API__

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
    HOST_TO_FIM,
    FIM_TO_HOST,
} FimMemcpyType;

typedef enum __FimOpType {
    OP_MAT_MUL,
    OP_ELT_ADD,
    OP_ELT_MUL,
    OP_BATCH_NORM,
    OP_RELU,
} FimOpType;

typedef enum __FimPrecision {
    FIM_FP16,
    FIM_INT8,
} FimPrecision;

#endif
