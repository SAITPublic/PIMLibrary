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
#ifndef _GPU_CUSTOM_OP_KERN_
#define _GPU_CUSTOM_OP_KERN_

#include "hip/hip_runtime.h"

struct rocblas_reduce_sum {
    template <typename T>
    __forceinline__ __device__ void operator()(T &__restrict__ a, const T &__restrict__ b)
    {
        a += b;
    }
};

template <int k, typename REDUCE, typename T>
struct rocblas_reduction_s {
    __forceinline__ __device__ void operator()(int tx, T *x)
    {
        // Reduce the lower half with the upper half
        if (tx < k) REDUCE{}(x[tx], x[tx + k]);
        __syncthreads();

        // Recurse down with k / 2
        rocblas_reduction_s<k / 2, REDUCE, T>{}(tx, x);
    }
};

// leaf node for terminating recursion
template <typename REDUCE, typename T>
struct rocblas_reduction_s<0, REDUCE, T> {
    __forceinline__ __device__ void operator()(int tx, T *x) {}
};

template <int NB, typename REDUCE, typename T>
__attribute__((flatten)) __device__ void rocblas_reduction(int tx, T *x)
{
    static_assert(NB > 1 && !(NB & (NB - 1)), "NB must be a power of 2");
    __syncthreads();
    rocblas_reduction_s<NB / 2, REDUCE, T>{}(tx, x);
}

template <int NB, typename T>
__attribute__((flatten)) __device__ void rocblas_sum_reduce(int tx, T *x)
{
    rocblas_reduction<NB, rocblas_reduce_sum>(tx, x);
}

#include "gpu_custom_addmv.gpuk"
#include "gpu_custom_gemv.gpuk"

#endif /* _GPU_CUSTOM_OP_KERN_ */
