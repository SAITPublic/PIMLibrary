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
    get_fim_block_info(&fbi_);
#ifdef EMULATOR
    fim_emulator_ = fim::runtime::emulator::FimEmulator::get_instance();
    fmtd_size_per_ch_ = 50000;
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

    max_crf_size_ = 128;
    max_crf_lut_size_ = 300;
    int max_srf_size = 2048;

    hipMalloc((void**)&d_crf_bin_lut_, (int)OP_DUMMY * max_crf_lut_size_ * max_crf_size_);
    h_crf_size_lut_ = (int*)malloc(sizeof(int) * max_crf_lut_size_ * (int)OP_DUMMY);
    create_crf_lut();

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

    fim_manager_->alloc_memory((void**)&fim_gemv_tmp_buffer_, 2 * 1024 * 1024, MEM_TYPE_FIM);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    hipFree((void*)d_crf_bin_lut_);
    hipFree((void*)d_srf_bin_buffer_);
    free(h_crf_size_lut_);
#ifdef EMULATOR
    hipFree((void*)d_fmtd16_);
    hipFree((void*)d_fmtd16_size_);
    free(h_fmtd16_);
    free(h_fmtd16_size_);
    free(h_fmtd32_);
    free(h_fmtd32_size_);
#endif
    fim_manager_->free_memory((void*)fim_gemv_tmp_buffer_, MEM_TYPE_FIM);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

void FimExecutor::create_crf_lut()
{
    uint8_t* temp_crf = new uint8_t[max_crf_size_];
    int crf_size = 0;

    int num_op_type = (int)OP_DUMMY;

    for (int j = 0; j < max_crf_lut_size_; j++) {
        fim_manager_->fim_crf_generator_->gen_binary_with_loop(OP_GEMV, j * 8 - 1, temp_crf, &crf_size);
        memcpy(d_crf_bin_lut_ + (int)OP_GEMV * max_crf_lut_size_ * max_crf_size_ + j * max_crf_size_, temp_crf,
               crf_size);
        h_crf_size_lut_[(int)OP_GEMV * max_crf_lut_size_ + j] = crf_size;
    }

    for (int i = 1; i < num_op_type; i++) {
        for (int j = 0; j < max_crf_lut_size_; j++) {
            fim_manager_->fim_crf_generator_->gen_binary_with_loop((FimOpType)i, j, temp_crf, &crf_size);
            memcpy(d_crf_bin_lut_ + i * max_crf_lut_size_ * max_crf_size_ + j * max_crf_size_, temp_crf, crf_size);
            h_crf_size_lut_[i * max_crf_lut_size_ + j] = crf_size;
        }
    }

    delete[] temp_crf;
}

int FimExecutor::get_loop_counter(FimOpType op_type, int input_size)
{
    int lc = 0;

    int num_transaction = (input_size / 16) / sizeof(uint16_t);
    int num_parallelism = fbi_.num_fim_blocks * fbi_.num_fim_chan * fbi_.num_fim_rank * fbi_.num_grf;
    int num_tile = num_transaction / num_parallelism;

    if (op_type == OP_RELU || op_type == OP_ELT_ADD || op_type == OP_ELT_MUL) {
        lc = num_tile / 2 - 1;
    } else if (op_type == OP_GEMV) {
        num_tile = ceil((double)num_transaction / (double)fbi_.num_grf);
        lc = fbi_.num_grf * ceil((double)num_tile / 2) - 1;
    } else {
        lc = num_tile - 1;
    }

    return lc;
}

int FimExecutor::execute_add(FimBo* output, FimBo* operand0, FimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;
    unsigned blocks = 64;
    unsigned threads_per_block = 16;
    int lc = get_loop_counter(OP_ELT_ADD, output->size);
    int crf_lut_offset = (int)OP_ELT_ADD * max_crf_lut_size_ * max_crf_size_ + lc * max_crf_size_;
    int crf_size = h_crf_size_lut_[(int)OP_ELT_ADD * max_crf_lut_size_ + lc];

    hipLaunchKernelGGL(elt_op_fim_64cu_16th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)operand0->data, (uint8_t*)operand1->data, (uint8_t*)g_fim_base_addr,
                       (uint8_t*)output->data, output->size,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_lut_ + crf_lut_offset, crf_size);

#ifdef EMULATOR
    hipStreamSynchronize(NULL);
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
#else
    if (block) hipStreamSynchronize(stream);
#endif

    return ret;
}

int FimExecutor::execute_mul(FimBo* output, FimBo* operand0, FimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;
    unsigned blocks = 64;
    unsigned threads_per_block = 16;

    int lc = get_loop_counter(OP_ELT_MUL, output->size);
    int crf_lut_offset = (int)OP_ELT_MUL * max_crf_lut_size_ * max_crf_size_ + lc * max_crf_size_;
    int crf_size = h_crf_size_lut_[(int)OP_ELT_MUL * max_crf_lut_size_ + lc];

    hipLaunchKernelGGL(elt_op_fim_64cu_16th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)operand0->data, (uint8_t*)operand1->data, (uint8_t*)g_fim_base_addr,
                       (uint8_t*)output->data, output->size,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_lut_ + crf_lut_offset, crf_size);

#ifdef EMULATOR
    hipStreamSynchronize(NULL);
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
#else
    if (block) hipStreamSynchronize(stream);
#endif
    return ret;
}

int FimExecutor::execute_gemv(FimBo* output, FimBo* operand0, FimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int is_gemv_add = 0;

    FimBo* input = operand0;
    FimBo* weight = operand1;
    unsigned blocks = fbi_.num_fim_chan;
    unsigned threads_per_block = 64;

    int in_size = weight->bshape.w;
    int out_size = weight->bshape.h;
    int real_out_size = weight->bshape_r.h;
    int n_batch = input->bshape.n;
    int n_in_tile = in_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_out_tile = out_size / (blocks * fbi_.num_fim_blocks * fbi_.num_grf_B);

    FIM_PROFILE_TICK(CreateCRFBin);
    int lc = (get_loop_counter(OP_GEMV, in_size * sizeof(half)) + 1) / 8;
    int crf_lut_offset = (int)OP_GEMV * max_crf_lut_size_ * max_crf_size_ + lc * max_crf_size_;
    int crf_size = h_crf_size_lut_[(int)OP_GEMV * max_crf_lut_size_ + lc];
    FIM_PROFILE_TOCK(CreateCRFBin);

    FIM_PROFILE_TICK(RunGemvKernel);
    hipLaunchKernelGGL(gemv_fim_64cu_64th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)g_fim_base_addr /* fim control base */, (uint8_t*)weight->data /* fim weight base */,
                       (uint8_t*)fim_gemv_tmp_buffer_, /* fim hw output buffer */
                       (uint8_t*)input->data, (uint8_t*)output->data, n_batch, n_in_tile, n_out_tile, real_out_size,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_lut_ + crf_lut_offset, crf_size, is_gemv_add);
#ifndef EMULATOR
    if (block) hipStreamSynchronize(stream);
    FIM_PROFILE_TOCK(RunGemvKernel);
#endif
#ifdef EMULATOR
    hipStreamSynchronize(NULL);
    FIM_PROFILE_TOCK(RunGemvKernel);

    FIM_PROFILE_TICK(RunGemvEmulation);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;

    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    fim_emulator_->execute_gemv(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV, g_fim_base_addr,
                                fim_gemv_tmp_buffer_);
    FIM_PROFILE_TOCK(RunGemvEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_gemv_add(FimBo* output, FimBo* operand0, FimBo* operand1, hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int is_gemv_add = 1;

    FimBo* input = operand0;
    FimBo* weight = operand1;
    unsigned blocks = fbi_.num_fim_chan;
    unsigned threads_per_block = 64;

    int in_size = weight->bshape.w;
    int out_size = weight->bshape.h;
    int real_out_size = weight->bshape_r.h;
    int n_batch = input->bshape.n;
    int n_in_tile = in_size * sizeof(uint16_t) / fbi_.trans_size / fbi_.num_grf_A;
    int n_out_tile = out_size / (blocks * fbi_.num_fim_blocks * fbi_.num_grf_B);

    FIM_PROFILE_TICK(CreateCRFBin);
    int lc = (get_loop_counter(OP_GEMV, in_size * sizeof(half)) + 1) / 8;
    int crf_lut_offset = (int)OP_GEMV * max_crf_lut_size_ * max_crf_size_ + lc * max_crf_size_;
    int crf_size = h_crf_size_lut_[(int)OP_GEMV * max_crf_lut_size_ + lc];
    FIM_PROFILE_TOCK(CreateCRFBin);

    FIM_PROFILE_TICK(RunGemvKernel);
    hipLaunchKernelGGL(gemv_fim_64cu_64th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)g_fim_base_addr /* fim control base */, (uint8_t*)weight->data /* fim weight base */,
                       (uint8_t*)fim_gemv_tmp_buffer_, /* fim hw output buffer */
                       (uint8_t*)input->data, (uint8_t*)output->data, n_batch, n_in_tile, n_out_tile, real_out_size,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_lut_ + crf_lut_offset, crf_size, is_gemv_add);
#ifndef EMULATOR
    if (block) hipStreamSynchronize(stream);
    FIM_PROFILE_TOCK(RunGemvKernel);
#endif
#ifdef EMULATOR
    hipStreamSynchronize(NULL);
    FIM_PROFILE_TOCK(RunGemvKernel);

    FIM_PROFILE_TICK(RunGemvEmulation);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;

    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    fim_emulator_->execute_gemv_add(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV, g_fim_base_addr,
                                    fim_gemv_tmp_buffer_);
    FIM_PROFILE_TOCK(RunGemvEmulation);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_relu(FimBo* output, FimBo* fim_data, hipStream_t stream, bool block)
{
    DLOG(INFO) << "called";
    int ret = 0;
    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    int lc = get_loop_counter(OP_RELU, output->size);
    int crf_lut_offset = (int)OP_RELU * max_crf_lut_size_ * max_crf_size_ + lc * max_crf_size_;
    int crf_size = h_crf_size_lut_[(int)OP_RELU * max_crf_lut_size_ + lc];

    hipLaunchKernelGGL(relu_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream,
                       (uint8_t*)fim_data->data, (uint8_t*)g_fim_base_addr, (uint8_t*)output->data, (int)output->size,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_lut_ + crf_lut_offset, crf_size);

#ifdef EMULATOR
    hipStreamSynchronize(NULL);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_RELU);
    fim_emulator_->execute_relu(output, fim_data, h_fmtd32_, h_fmtd32_size_[0], g_fim_base_addr);
#else
    if (block) hipStreamSynchronize(stream);
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
                            double epsilon, hipStream_t stream, bool block)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    /* TODO: modify 128 to crf size */

    int lc = get_loop_counter(OP_BN, output->size);
    int crf_lut_offset = (int)OP_BN * max_crf_lut_size_ * max_crf_size_ + lc * max_crf_size_;
    int crf_size = h_crf_size_lut_[(int)OP_BN * max_crf_lut_size_ + lc];

    uint8_t* srf_binary = new uint8_t[fbi_.num_fim_chan * fbi_.num_fim_rank * fbi_.trans_size];
    int srf_size = fbi_.num_fim_chan * fbi_.num_fim_rank * fbi_.trans_size;
    preprocess_srf(beta, gamma, mean, variance, epsilon, srf_binary);

    hipMemcpy((void*)d_srf_bin_buffer_, (void*)srf_binary, srf_size, hipMemcpyHostToDevice);
    hipMemcpy((void*)g_fim_base_addr, fim_data->data, fim_data->size, hipMemcpyHostToDevice);

    // printf("crf_size:%d, srf_size:%d, output->size:%d\n", crf_size, srf_size, output->size);
    // printf("bshaped(%d,%d,%d,%d)\n", output->bshape.w, output->bshape.h, output->bshape.c, output->bshape.n);

    hipLaunchKernelGGL(bn_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, stream, (uint8_t*)g_fim_base_addr,
                       (uint8_t*)g_fim_base_addr, (uint8_t*)output->data, (int)output->size, output->bshape.n,
                       output->bshape.c, output->bshape.w,
#ifdef EMULATOR
                       (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
#endif
                       (uint8_t*)d_crf_bin_lut_ + crf_lut_offset, crf_size, (uint8_t*)d_srf_bin_buffer_, srf_size);

#ifdef EMULATOR
    hipStreamSynchronize(NULL);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_BN);
    fim_emulator_->execute_bn(output, fim_data, h_fmtd32_, h_fmtd32_size_[0]);
#else
    if (block) hipStreamSynchronize(stream);
#endif
    delete[] srf_binary;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_sync() { hipStreamSynchronize(NULL); }
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
