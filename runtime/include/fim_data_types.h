/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_DATA_TYPE_H_
#define _FIM_DATA_TYPE_H_

#include <stddef.h>
#include <stdint.h>

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

typedef enum __FimMemFlag { ELT_OP, GEMV_INPUT, GEMV_WEIGHT, GEMV_OUTPUT } FimMemFlag;

typedef enum __FimMemCpyType {
    HOST_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_FIM,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    DEVICE_TO_FIM,
    FIM_TO_HOST,
    FIM_TO_DEVICE,
    FIM_TO_FIM,
} FimMemCpyType;

typedef enum __FimOpType {
    OP_GEMV,
    OP_ELT_ADD,
    OP_ELT_MUL,
    OP_RELU,
    OP_BN,
    OP_DUMMY,
} FimOpType;

typedef enum __FimPrecision {
    FIM_FP16,
    FIM_INT8,
} FimPrecision;

typedef struct __FimBShape {
    uint32_t w;
    uint32_t h;
    uint32_t c;
    uint32_t n;
} FimBShape;

typedef struct __FimBufferObject {
    FimMemType mem_type;
    FimBShape bshape;
    FimBShape bshape_r;
    FimPrecision precision;
    size_t size;
    void* data;
} FimBo;

typedef struct __FimDescriptor {
    FimBShape bshape;
    FimBShape bshape_r;
    FimPrecision precision;
    FimOpType op_type;
} FimDesc;
typedef struct __FimGemvBundle {
    FimBo* in;
    FimBo* wei;
    FimBo* out;
} FimGemvBundle;

#endif /* _FIM_DATA_TYPE_H_ */
