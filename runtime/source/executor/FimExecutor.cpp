#include "executor/FimExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "executor/fim_hip_kernels/fim_op_kernels.fimk"
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"
#include "utility/fim_util.h"

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
    DLOG(INFO) << "Intialization done ";

    int ret = 0;
    hipGetDeviceProperties(&devProp_, 0);
    std::cout << " System minor " << devProp_.minor << std::endl;
    std::cout << " System major " << devProp_.major << std::endl;
    std::cout << " agent prop name " << devProp_.name << std::endl;
    std::cout << " hip Device prop succeeded " << std::endl;

    /* TODO: get fim control base address from device driver */
    /* roct should write fim_base_va */
    FILE* fp;
    fp = fopen("fim_base_va.txt", "rt");
    fscanf(fp, "%lX", &fimBaseAddr_);
    printf("fimBaseAddr_ : 0x%X\n");
    fclose(fp);

    get_fim_block_info(&fbi_);

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
    } else {
        /* todo:implement other operation function */
        hipLaunchKernelGGL(dummy_kernel, dim3(1), dim3(1), 0, 0);
        return -1;
    }
    hipStreamSynchronize(NULL);

    return ret;
}

int FimExecutor::Execute(FimBo* output, FimBo* fimData, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;
    size_t size = output->size;
    uint8_t* fimBasePtr = (uint8_t*)fimBaseAddr_;
    uint8_t* outputPtr = (uint8_t*)output->data;

    if (opType == OP_ELT_ADD) {
        std::cout << "fimBaseAddr = " << fimBaseAddr_ << std::endl;
        hipMemcpy((void*)fimBasePtr, fimData->data, fimData->size, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(elt_add_fim_1cu_1th_fp16, dim3(1), dim3(1), 0, 0, (uint8_t*)fimBasePtr, (uint8_t*)fimBasePtr,
                           (uint8_t*)outputPtr, (int)output->size);

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
