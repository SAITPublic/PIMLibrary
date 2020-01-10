#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "FimRuntime.h"
#include "fim_data_types.h"
#include "hip/hip_runtime.h"
#include "executor/hip_fim_kernel/fim_op_kernels.fimk"

namespace fim {
namespace runtime {

FimRuntime::FimRuntime(FimRuntimeType rtType)
    :rtType_(rtType), threadCnt_(16), blockCnt_(64)
{
    std::cout << "fim::runtime creater call" << std::endl;
}

int FimRuntime::Initialize(void)
{
    std::cout << "fim::runtime Initialize call" << std::endl;

    int ret = 0;
    if (rtType_ == RT_TYPE_HIP) {
        hipGetDeviceProperties(&devProp_, 0);
        std::cout << " System minor " << devProp_.minor << std::endl;
        std::cout << " System major " << devProp_.major << std::endl;
        std::cout << " agent prop name " << devProp_.name << std::endl;
        std::cout << " hip Device prop succeeded " << std::endl;
    }

    return ret;
}

int FimRuntime::Deinitialize(void)
{
    std::cout << "fim::runtime Deinitialize call" << std::endl;

    int ret = 0;
    return ret;
}

int FimRuntime::AllocMemory(float** ptr, size_t size, FimMemType memType)
{
    std::cout << "fim::runtime AllocMemory call" << std::endl;

    int ret = 0;

    if (rtType_ == RT_TYPE_HIP) {
        if (memType == MEM_TYPE_DEVICE) {
            if (hipMalloc((void**)ptr, size) != hipSuccess) {
                return -1;
            }
        }
        else if (memType == MEM_TYPE_HOST) {
            *ptr = (float*)malloc(size);
        }
        else if (memType == MEM_TYPE_FIM) {
            /* todo:implement fimalloc function */
        }
    }

    return ret;
}

int FimRuntime::FreeMemory(float* ptr, FimMemType memType)
{
    std::cout << "fim::runtime FreeMemory call" << std::endl;

    int ret = 0;

    if (rtType_ == RT_TYPE_HIP) {
        if (memType == MEM_TYPE_DEVICE) {
            if (hipFree(ptr) != hipSuccess) {
                return -1;
            }
        }
        else if (memType == MEM_TYPE_HOST) {
            free(ptr);
        }
        else if (memType == MEM_TYPE_FIM) {
            /* todo:implement fimfree function */
        }
    }

    return ret;
}

int FimRuntime::CopyMemory(float* dst, float* src, size_t size, FimMemcpyType cpyType)
{
    std::cout << "fim::runtime Memcpy call" << std::endl;

    int ret = 0;

    if (rtType_ == RT_TYPE_HIP) {
        if (cpyType == HOST_TO_FIM) {
            if (hipMemcpy(dst, src, size, hipMemcpyHostToDevice)!= hipSuccess) {
                return -1;
            }
        }
        else if (cpyType == FIM_TO_HOST) {
            if (hipMemcpy(dst, src, size, hipMemcpyDeviceToHost)!= hipSuccess) {
                return -1;
            }
        }
    }

    return ret;
}

int FimRuntime::Execute(float* output, float* operand0, float* operand1, size_t size, FimOpType opType, FimPrecision precision)
{
    std::cout << "fim::runtime Execute call" << std::endl;

    int ret = 0;

    if (rtType_ == RT_TYPE_HIP) {
        if (opType == OP_ELT_ADD) {
            hipLaunchKernelGGL(
                    eltwise_add_fp32,
                    dim3(size / threadCnt_),
                    dim3(threadCnt_),
                    0,
                    0,
                    operand0,
                    operand1,
                    output);
        }
        else {
            /* todo:implement other operation function */
            return -1;
        }
    }

    return ret;
}

} /* namespace runtime */
} /* namespace fim */

