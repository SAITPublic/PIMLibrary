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

typedef struct __FimMemory {
    size_t size;
    FimMemType memType;
    void* ptr;
} FimMemory;

#endif
