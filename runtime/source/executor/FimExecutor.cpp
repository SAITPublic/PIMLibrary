#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "hip/hip_runtime.h"
#include "executor/fim_hip_kernels/fim_op_kernels.fimk"
#include "executor/FimExecutor.h"

namespace fim {
namespace runtime {
namespace executor {

FimExecutor::FimExecutor(FimRuntimeType rtType, FimPrecision precision)
    : rtType_(rtType), precision_(precision), threadCnt_(16)
{
    std::cout << "fim::runtime::executor creator call" << std::endl;
}

FimExecutor* FimExecutor::getInstance(FimRuntimeType rtType, FimPrecision precision)
{
    std::cout << "fim::runtime::executor getInstance call" << std::endl;

    static FimExecutor* instance_ = new FimExecutor(rtType, precision);

    return instance_;
}

int FimExecutor::Initialize(void)
{
    std::cout << "fim::runtime::executor Initialize call" << std::endl;

    int ret = 0;

    hipGetDeviceProperties(&devProp_, 0);
    std::cout << " System minor " << devProp_.minor << std::endl;
    std::cout << " System major " << devProp_.major << std::endl;
    std::cout << " agent prop name " << devProp_.name << std::endl;
    std::cout << " hip Device prop succeeded " << std::endl;

    return ret;
}

int FimExecutor::Deinitialize(void)
{
    std::cout << "fim::runtime::executor Deinitialize call" << std::endl;

    int ret = 0;

    return ret;
}

int FimExecutor::Execute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType)
{
    std::cout << "fim::runtime::executor Execute call" << std::endl;

    int ret = 0;

    if (opType == OP_ELT_ADD) {
        if (precision_ == FIM_FP16) {
            hipLaunchKernelGGL(
                eltwise_add_fp16,
                dim3(size / threadCnt_),
                dim3(threadCnt_),
                0,
                0,
                (__half*)operand0,
                (__half*)operand1,
                (__half*)output);
        }
        else if (precision_  == FIM_INT8) {
            hipLaunchKernelGGL(
                eltwise_add_int8,
                dim3(size / threadCnt_),
                dim3(threadCnt_),
                0,
                0,
                (char*)operand0,
                (char*)operand1,
                (char*)output);
        }
    }
    else {
        /* todo:implement other operation function */
        return -1;
    }

    return ret;
}

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */
