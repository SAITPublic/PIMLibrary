#include "executor/FimExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "executor/fim_hip_kernels/fim_op_kernels.fimk"
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"
#include "utility/fim_util.h"

#define BLOCKS 32
#define WIDTH 370
#define MT_NUM (BLOCKS * WIDTH)

namespace fim
{
namespace runtime
{
namespace executor
{
FimExecutor::FimExecutor(FimRuntimeType rt_type, FimPrecision precision)
    : rt_type_(rt_type), precision_(precision), thread_cnt_(16)
{
    DLOG(INFO) << "called ";

#ifdef EMULATOR
    fim_emulator_ = fim::runtime::emulator::FimEmulator::get_instance();
#endif
}

FimExecutor* FimExecutor::get_instance(FimRuntimeType rt_type, FimPrecision precision)
{
    DLOG(INFO) << "Called";
    static FimExecutor* instance_ = new FimExecutor(rt_type, precision);

    return instance_;
}

int FimExecutor::initialize(void)
{
    DLOG(INFO) << "Intialization done ";

    int ret = 0;
    hipGetDeviceProperties(&dev_prop_, 0);
    std::cout << " System minor " << dev_prop_.minor << std::endl;
    std::cout << " System major " << dev_prop_.major << std::endl;
    std::cout << " agent prop name " << dev_prop_.name << std::endl;
    std::cout << " hip Device prop succeeded " << std::endl;

#ifdef EMULATOR
    /*
        int reserved_size = 1024 * 1024 * 1024;
        hipMalloc((void**)&fim_base_addr_, reserved_size);
        fmtd16_ = (FimMemTraceData*)(fim_base_addr_ + 256 * 1024 * 1024);
        fmtd32_ = (FimMemTraceData*)(fim_base_addr_ + 512 * 1024 * 1024);
        hipMalloc((void**)&fmtd16_size_, sizeof(int));
    */
    size_t n_ = 4;
    size_t nbytes_ = n_ * sizeof(uint64_t);

    cbin_h_ = (uint64_t*)malloc(nbytes_);
    cmode_h_ = (uint64_t*)malloc(nbytes_);

    cbin_h_[0] = 0x1b18800098800000;
    cbin_h_[1] = 0xe000000400000007;
    cbin_h_[2] = 0x00000000f0000000;
    cbin_h_[3] = 0x0000000000000000;
    cmode_h_[0] = 0x0000000000000001;
    cmode_h_[1] = 0x0000000000000000;
    cmode_h_[2] = 0x0000000000000000;
    cmode_h_[3] = 0x0000000000000000;

    fmtd16_size_ = MT_NUM * sizeof(FimMemTraceData);
    fmtd16_ = (FimMemTraceData*)malloc(fmtd16_size_);
    fmtd_h_ = (FimMemTraceData*)malloc(fmtd16_size_);

    for (size_t i = 0; i < MT_NUM; i++) {
        fmtd_h_[i].data[0] = 0x0;
        fmtd_h_[i].data[1] = 0x0;
        fmtd_h_[i].addr = 0x0;
        fmtd_h_[i].block_id = 0x0;
        fmtd_h_[i].thread_id = 0x0;
        fmtd_h_[i].cmd = '\0';
    }

    hipMalloc(&cmode_d_, nbytes_);
    hipMalloc(&cbin_d_, nbytes_);
    hipMalloc(&fmtd_d_, fmtd16_size_);

    hipMemcpy(cmode_d_, cmode_h_, nbytes_, hipMemcpyHostToDevice);
    hipMemcpy(cbin_d_, cbin_h_, nbytes_, hipMemcpyHostToDevice);
    hipMemcpy(fmtd_d_, fmtd_h_, fmtd16_size_, hipMemcpyHostToDevice);
#endif
    /* TODO: get fim control base address from device driver */
    /* roct should write fim_base_va */
    FILE* fp;
    fp = fopen("fim_base_va.txt", "rt");
    fscanf(fp, "%lX", &fim_base_addr_);
    printf("fim_base_addr_ : 0x%lX\n", fim_base_addr_);
    fclose(fp);

    return ret;
}

int FimExecutor::deinitialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

#ifdef EMULATOR
    hipFree(cmode_d_);
    hipFree(cbin_d_);
    hipFree(fmtd_d_);

    free(cmode_h_);
    free(cbin_h_);
    free(fmtd_h_);
    free(fmtd16_);
#endif

    return ret;
}

int FimExecutor::execute(void* output, void* operand0, void* operand1, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (op_type == OP_ELT_ADD) {
    } else {
        /* todo:implement other operation function */
        return -1;
    }
    hipStreamSynchronize(NULL);

    return ret;
}

int FimExecutor::execute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;
    size_t size = output->size;

    if (op_type == OP_ELT_ADD) {
    } else {
        /* todo:implement other operation function */
        hipLaunchKernelGGL(dummy_kernel, dim3(1), dim3(1), 0, 0);
        return -1;
    }
    hipStreamSynchronize(NULL);

    return ret;
}

int FimExecutor::execute(FimBo* output, FimBo* fim_data, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (op_type == OP_ELT_ADD) {
        hipMemcpy((void*)fim_base_addr_, fim_data->data, fim_data->size, hipMemcpyHostToDevice);
#ifdef EMULATOR
        const unsigned blocks_ = BLOCKS;
        const unsigned threads_per_block_ = 16;
        const unsigned input_size = blocks_ * 1024;
        hipLaunchKernelGGL(elt_add_fim_emul, dim3(blocks_), dim3(threads_per_block_), 0, 0, (uint8_t*)fim_base_addr_,
                           (uint8_t*)cmode_d_, (uint8_t*)cbin_d_, 0, input_size, fmtd_d_, WIDTH);
#else
        hipLaunchKernelGGL(elt_add_fim_1cu_1th_fp16, dim3(1), dim3(1), 0, 0, (uint8_t*)fim_base_addr_,
                           (uint8_t*)fim_base_addr_, (uint8_t*)output->data, (int)output->size,
                           (FimMemTraceData*)fmtd16_, (int*)fmtd16_size_);
#endif
    } else {
        /* todo:implement other operation function */
        hipLaunchKernelGGL(dummy_kernel, dim3(1), dim3(1), 0, 0);
        return -1;
    }
    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy(fmtd16_, fmtd_d_, fmtd16_size_, hipMemcpyDeviceToHost);

    FILE* fp2;
    fp2 = fopen("out.txt", "w");

    for (size_t i = 0; i < BLOCKS; i++) {
        for (size_t j = 0; j < WIDTH; j++) {
            int idx = i * WIDTH + j;
            fprintf(fp2, "[memt] Block: %d Thread: %d Addr: %lx Data: %lx%lx Cmd: %c\n", fmtd16_[idx].block_id,
                    fmtd16_[idx].thread_id, fmtd16_[idx].addr, fmtd16_[idx].data[1], fmtd16_[idx].data[0],
                    fmtd16_[idx].cmd);
        }
    }
    fclose(fp2);

    fim_emulator_->convert_mem_trace_from_16B_to_32B(fmtd32_, fmtd16_, fmtd16_size_);
    fim_emulator_->execute_fim(output, fim_data, fmtd32_, op_type);
#endif

    return ret;
}

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */
