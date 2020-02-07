#include "executor/FimExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "executor/fim_hip_kernels/fim_op_kernels.fimk"
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
namespace executor
{
FimExecutor::FimExecutor(FimRuntimeType rtType, FimPrecision precision)
    : rtType_(rtType), precision_(precision), threadCnt_(16)
{
    DLOG(INFO) << "called ";
}

FimExecutor* FimExecutor::getInstance(FimRuntimeType rtType, FimPrecision precision)
{
    DLOG(INFO) << "Called";
    static FimExecutor* instance_ = new FimExecutor(rtType, precision);

    return instance_;
}

int FimExecutor::Initialize(void)
{
    google::InitGoogleLogging("FIMExecutor");
    DLOG(INFO) << "Intialization done ";

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
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

int FimExecutor::Execute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (opType == OP_ELT_ADD) {
        if (precision_ == FIM_FP16) {
            hipLaunchKernelGGL(eltwise_add_fp16, dim3(size / threadCnt_), dim3(threadCnt_), 0, 0, (__half*)operand0,
                               (__half*)operand1, (__half*)output);
        } else if (precision_ == FIM_INT8) {
            hipLaunchKernelGGL(eltwise_add_int8, dim3(size / threadCnt_), dim3(threadCnt_), 0, 0, (char*)operand0,
                               (char*)operand1, (char*)output);
        }
    } else {
        /* todo:implement other operation function */
        return -1;
    }
    hipStreamSynchronize(NULL);

    return ret;
}

int FimExecutor::Execute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;
    size_t size = output->size;

    if (opType == OP_ELT_ADD) {
        if (precision_ == FIM_FP16) {
            hipLaunchKernelGGL(eltwise_add_fp16, dim3(size / threadCnt_), dim3(threadCnt_), 0, 0,
                               (__half*)operand0->data, (__half*)operand1->data, (__half*)output->data);
        } else if (precision_ == FIM_INT8) {
            hipLaunchKernelGGL(eltwise_add_int8, dim3(size / threadCnt_), dim3(threadCnt_), 0, 0, (char*)operand0->data,
                               (char*)operand1->data, (char*)output->data);
        }
    } else if (opType == OP_GEMV) {
        if (precision_ == FIM_FP16) {
            size_t in_size = operand0->size / sizeof(__half);
            size_t out_size = output->size / sizeof(__half);
            const unsigned int blocks = 64;
            const unsigned int threads_per_block = 1;
            const unsigned int loop_cnt = out_size / blocks;

            hipLaunchKernelGGL(gemv_64cu_1th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (__half*)operand0->data,
                               (__half*)operand1->data, (__half*)output->data, in_size, loop_cnt);
        }
    } else {
        /* todo:implement other operation function */
        hipLaunchKernelGGL(dummy_kernel, dim3(1), dim3(1), 0, 0);
        return -1;
    }
    hipStreamSynchronize(NULL);

    return ret;
}

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */
