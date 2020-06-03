#include "executor/FimExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "executor/fim_hip_kernels/fim_op_kernels.fimk"
#include "hip/hip_runtime.h"
#include "utility/fim_dump.hpp"
#include "utility/fim_log.h"
#include "utility/fim_profile.h"
#include "utility/fim_util.h"

extern uint64_t g_fim_base_addr;
namespace fim
{
namespace runtime
{
namespace executor
{
FimExecutor::FimExecutor(FimRuntimeType rt_type, FimPrecision precision)
    : rt_type_(rt_type), precision_(precision), thread_cnt_(16)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";

#ifdef EMULATOR
    fim_emulator_ = fim::runtime::emulator::FimEmulator::get_instance();

    get_fim_block_info(&fbi_);
    fmtd_size_per_ch_ = 20000;
    max_block_size_ = fbi_.num_fim_chan;
    max_fmtd_size_ = fmtd_size_per_ch_ * max_block_size_;
#endif
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

FimExecutor* FimExecutor::get_instance(FimRuntimeType rt_type, FimPrecision precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " Called";
    static FimExecutor* instance_ = new FimExecutor(rt_type, precision);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return instance_;
}

int FimExecutor::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " Intialization done ";

    int ret = 0;
    hipGetDeviceProperties(&dev_prop_, 0);
    DLOG(INFO) << " System minor " << dev_prop_.minor << std::endl;
    DLOG(INFO) << " System major " << dev_prop_.major << std::endl;
    DLOG(INFO) << " agent prop name " << dev_prop_.name << std::endl;
    DLOG(INFO) << " hip Device prop succeeded " << std::endl;

    fim_manager_ = fim::runtime::manager::FimManager::get_instance(rt_type_, precision_);

    int max_crf_size = 128;
    int max_srf_size = 2048;
    hipMalloc((void**)&d_crf_bin_buffer_, max_crf_size);
    hipMalloc((void**)&d_srf_bin_buffer_, max_srf_size);
#ifdef EMULATOR
    int reserved_fmtd_size = max_fmtd_size_ * sizeof(FimMemTraceData);
    hipMalloc((void**)&d_fmtd16_, reserved_fmtd_size);
    hipMalloc((void**)&d_fmtd16_size_, sizeof(int));

    h_fmtd16_ = (FimMemTraceData*)malloc(reserved_fmtd_size);
    h_fmtd32_ = (FimMemTraceData*)malloc(reserved_fmtd_size);
    h_fmtd16_size_ = (int*)malloc(sizeof(int));
    h_fmtd32_size_ = (int*)malloc(sizeof(int));
#endif
    /* FIM HW can generate only gemv output without reduction sum */
    /* so FimExecutor needs to maintain intermediate output buffer for gemv op */
    hipMalloc((void**)&fim_gemv_tmp_buffer_, 2 * 1024 * 1024);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    hipFree((void*)d_crf_bin_buffer_);
    hipFree((void*)d_srf_bin_buffer_);
#ifdef EMULATOR
    hipFree((void*)d_fmtd16_);
    hipFree((void*)d_fmtd16_size_);
    free(h_fmtd16_);
    free(h_fmtd16_size_);
    free(h_fmtd32_);
    free(h_fmtd32_size_);
#endif
    hipFree((void*)fim_gemv_tmp_buffer_);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_add(FimBo* output, FimBo* operand0, FimBo* operand1)
{
    DLOG(INFO) << "called";
    int ret = 0;
    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    fim_manager_->create_crf_binary(OP_ELT_ADD, output->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(elt_op_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)operand0->data,
                       (uint8_t*)operand1->data, (uint8_t*)g_fim_base_addr, (uint8_t*)output->data, output->size,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_buffer_, crf_size);

    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                     OP_ELT_ADD);
    fim_emulator_->execute_elt_op(output, operand0, operand1, h_fmtd32_, h_fmtd32_size_[0], g_fim_base_addr);
#endif

    return ret;
}

int FimExecutor::execute_mul(FimBo* output, FimBo* operand0, FimBo* operand1)
{
    DLOG(INFO) << "called";
    int ret = 0;
    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    fim_manager_->create_crf_binary(OP_ELT_MUL, output->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(elt_op_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)operand0->data,
                       (uint8_t*)operand1->data, (uint8_t*)g_fim_base_addr, (uint8_t*)output->data, output->size,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_buffer_, crf_size);

    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                     OP_ELT_MUL);
    fim_emulator_->execute_elt_op(output, operand0, operand1, h_fmtd32_, h_fmtd32_size_[0], g_fim_base_addr);
#endif

    return ret;
}

int FimExecutor::execute_gemv(FimBo* output, FimBo* operand0, FimBo* operand1)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    FimBo* input = operand0;
    FimBo* weight = operand1;

    int in_size = weight->bshape.w;
    int out_size = weight->bshape.h;
    int num_batch = input->bshape.n;

    FIM_PROFILE_TICK(CreateCRFBin);
    fim_manager_->create_crf_binary(OP_GEMV, in_size * sizeof(half), out_size * sizeof(half));
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();
    FIM_PROFILE_TOCK(CreateCRFBin);

    // FIXME : change 128 to a meaningful variable.
    FIM_PROFILE_TICK(CopyCRFBin);
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);
    FIM_PROFILE_TOCK(CopyCRFBin);

    FIM_PROFILE_TICK(RunGemvKernel);
    for (int iter = 0; iter < 1; iter++) {
        hipLaunchKernelGGL(
            gemv_fim_1cu_2th_fp16, dim3(1), dim3(2), 0, 0, (uint8_t*)g_fim_base_addr /* fim control base */,
            (uint8_t*)weight->data /* fim weight base */, (uint8_t*)fim_gemv_tmp_buffer_, /* fim hw output buffer */
            (uint8_t*)input->data, (uint8_t*)output->data, in_size, num_batch, out_size,
#ifdef EMULATOR
            (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_,
#endif
            (uint8_t*)d_crf_bin_buffer_, crf_size);
    }
    hipStreamSynchronize(NULL);
    FIM_PROFILE_TOCK(RunGemvKernel);

#ifdef EMULATOR
    FIM_PROFILE_TICK(RunGemvEmulation);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    fim_emulator_->execute_gemv(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV);
    FIM_PROFILE_TOCK(RunGemvEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_relu(FimBo* output, FimBo* fim_data)
{
    DLOG(INFO) << "called";
    int ret = 0;
    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    fim_manager_->create_crf_binary(OP_RELU, output->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(relu_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)fim_data->data,
                       (uint8_t*)g_fim_base_addr, (uint8_t*)output->data, (int)output->size,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_buffer_, crf_size);
    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_RELU);
    fim_emulator_->execute_relu(output, fim_data, h_fmtd32_, h_fmtd32_size_[0], g_fim_base_addr);
#endif

    return ret;
}

int FimExecutor::preprocess_srf(FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance, double epsilon,
                                uint8_t* srf_binary)
{
    int num_fim_rank = fbi_.num_fim_rank;
    int num_fim_chan = fbi_.num_fim_chan;

    int cidx = 0;
    int rank = 0;
    int burst_idx = 0;
    int num_stride_reg = 2;
    int num_half_per_reg = 16;

    half* h_srf_binary = reinterpret_cast<half*>(srf_binary);
    half* h_beta = (half*)beta->data;
    half* h_gamma = (half*)gamma->data;
    half* h_var = (half*)variance->data;
    half* h_mean = (half*)mean->data;

    for (int ch_model = 0; ch_model < beta->bshape.c; ch_model++) {
        h_srf_binary[cidx * num_fim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx] =
            1 / sqrt((float)h_var[ch_model] + epsilon);  // scale
        h_srf_binary[cidx * num_fim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 1] =
            h_gamma[ch_model];  // gamma
        h_srf_binary[cidx * num_fim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 8] =
            -(float)h_mean[ch_model] / sqrt((float)h_var[ch_model] + epsilon);  // shift
        h_srf_binary[cidx * num_fim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 9] =
            h_beta[ch_model];  // beta
        rank++;

        if (rank >= num_fim_rank) {
            rank = 0;
            cidx++;
        }
        if (cidx >= num_fim_chan) {
            cidx = 0;
            burst_idx += num_stride_reg;
        }
        if (burst_idx >= 8) {
            std::cout << "error: this is not defined" << std::endl;
        }
    }
}

int FimExecutor::execute_bn(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance,
                            double epsilon)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    /* TODO: modify 128 to crf size */

    fim_manager_->create_crf_binary(OP_BN, output->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();
    uint8_t* srf_binary = new uint8_t[fbi_.num_fim_chan * fbi_.num_fim_rank * fbi_.trans_size];
    int srf_size = fbi_.num_fim_chan * fbi_.num_fim_rank * fbi_.trans_size;
    preprocess_srf(beta, gamma, mean, variance, epsilon, srf_binary);

    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);
    hipMemcpy((void*)d_srf_bin_buffer_, (void*)srf_binary, srf_size, hipMemcpyHostToDevice);
    hipMemcpy((void*)g_fim_base_addr, fim_data->data, fim_data->size, hipMemcpyHostToDevice);

    printf("crf_size:%d, srf_size:%d, output->size:%d\n", crf_size, srf_size, output->size);
    printf("bshaped(%d,%d,%d,%d)\n", output->bshape.w, output->bshape.h, output->bshape.c, output->bshape.n);

    hipLaunchKernelGGL(bn_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)g_fim_base_addr,
                       (uint8_t*)g_fim_base_addr, (uint8_t*)output->data, (int)output->size, output->bshape.n,
                       output->bshape.c, output->bshape.w,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_buffer_, crf_size, (uint8_t*)d_srf_bin_buffer_, srf_size);

    hipStreamSynchronize(NULL);
    delete[] srf_binary;

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_BN);
    fim_emulator_->execute_fim(output, fim_data, h_fmtd32_, h_fmtd32_size_[0], OP_BN);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_dummy(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    hipLaunchKernelGGL(dummy_kernel, dim3(1), dim3(1), 0, 0);
    hipStreamSynchronize(NULL);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */
