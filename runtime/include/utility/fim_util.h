#ifndef _FIM_UTIL_H_
#define _FIM_UTIL_H_

#include "executor/fim_hip_kernels/fim_crf_bins.h"
#include "fim_data_types.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "manager/FimInfo.h"
#include "utility/fim_log.h"

/* TODO: get VEGA20 scheme from device driver */
FimBlockInfo vega20_fbi = {
    .fim_addr_map = AMDGPU_VEGA20,
    .num_banks = 16,
    .num_bank_groups = 4,
    .num_rank_bit = 1,
    .num_row_bit = 14,
    .num_col_high_bit = 3,
    .num_bank_high_bit = 1,
    .num_bankgroup_bit = 2,
    .num_bank_low_bit = 1,
    .num_chan_bit = 6,
    .num_col_low_bit = 2,
    .num_offset_bit = 5,
    .num_grf = 8,
    .num_grf_A = 8,
    .num_grf_B = 8,
    .num_srf = 4,
    .num_col = 128,
    .num_row = 16384,
    .bl = 4,
    .num_fim_blocks = 8,
    .num_fim_rank = 1,
    .num_fim_chan = 64,
    .trans_size = 32,
    .num_out_per_grf = 16,
};

__host__ void get_fim_block_info(FimBlockInfo* fbi);
__device__ void reduce_sum_for_gemv_gpu(void* out, void* in, int out_size, int reduce_size);
__host__ __device__ uint32_t mask_by_bit(uint32_t value, uint32_t start, uint32_t end);
__host__ __device__ uint64_t addr_gen(uint32_t chan, uint32_t rank, uint32_t bankgroup, uint32_t bank, uint32_t row,
                                      uint32_t col);
__host__ __device__ uint64_t addr_gen_safe(uint32_t chan, uint32_t rank, uint32_t bg, uint32_t bank, uint32_t& row,
                                           uint32_t& col);
size_t get_aligned_size(FimDesc* fim_desc, FimMemFlag mem_flag, FimBo* fim_bo);
void pad_data(void* input, int in_size, int in_nsize, int batch_size, FimMemFlag mem_flag);
void pad_data(void* input, FimDesc* fim_desc, FimMemType mem_type, FimMemFlag mem_flag);
void align_shape(FimDesc* fim_desc, FimOpType op_type);

#ifdef EMULATOR
extern uint64_t g_fba;
extern int g_ridx[64];
extern int g_idx[64];
extern int m_width;
extern FimMemTraceData* g_fmtd16;

__device__ void record(int bid, char mtype, uint64_t paddr);
#endif

__device__ void GEN_WRITE_CMD(volatile uint8_t* __restrict__ dst, volatile uint8_t* __restrict__ src);
__device__ void GEN_READ_CMD(volatile uint8_t* __restrict__ dst, volatile uint8_t* __restrict__ src,
                             bool is_output = false);
__device__ void GEN_BLOCK_CMD(int type = 0);
__device__ void BLOCK_SYNC(int cu_ch_idx = 0, bool block_all_chan = true);
__device__ void R_CMD(volatile uint8_t* __restrict__ addr);
__device__ void W_CMD(volatile uint8_t* __restrict__ addr);
__device__ void W_CMD_R(volatile uint8_t* __restrict__ addr, volatile uint8_t* __restrict__ src);

__device__ void B_CMD(int type);

/* 1CU 2TH functions */

__device__ void add_transaction_all_1cu_2th(volatile uint8_t* __restrict__ fim_addr, bool is_write, uint32_t bg,
                                            uint32_t bank, uint32_t row, uint32_t col, uint8_t* burst, uint64_t offset,
                                            int loop_cnt = 1);
__device__ void change_fim_mode_1cu_2th(volatile uint8_t* __restrict__ fim_ctr, FimMode mode1, FimMode mode2,
                                        uint8_t* change_mode_bin, uint64_t offset);
__device__ void park_1cu_2th(volatile uint8_t* __restrict__ fim_addr, uint64_t offset);
__device__ void program_crf_1cu_2th(volatile uint8_t* __restrict__ fim_ctr, uint8_t* crf_bin, uint32_t cmd_size,
                                    uint64_t offset);
__device__ void program_srf_1cu_2th(volatile uint8_t* __restrict__ fim_ctr, uint8_t* srf_bin, uint32_t srf_bin_size,
                                    uint64_t offset);
__device__ void compute_relu_1cu_2th(volatile uint8_t* __restrict__ fim_output, volatile uint8_t* __restrict__ fim_data,
                                     int num_tile, uint64_t offset);
__device__ void compute_bn_1cu_2th(volatile uint8_t* __restrict__ fim_data, int num_tile, uint64_t offset);
__device__ int get_num_tile(int dim);
__device__ int get_result_col(int dim);
__device__ int gemv_get_result_col(int input_dim, int output_dim, int num_in_tile, int num_out_tile);

__device__ void read_result_bn_1cu_2th(volatile uint8_t* __restrict__ output, volatile uint8_t* __restrict__ fim_data,
                                       int num_batch, int num_ch, int num_width, uint32_t s_row, uint32_t s_col,
                                       uint64_t offset);
__device__ void read_result_bn_64cu_2th(volatile uint8_t* __restrict__ output, volatile uint8_t* __restrict__ fim_data,
                                        int num_batch, int num_ch, int num_width, uint32_t s_row, uint32_t s_col,
                                        uint64_t offset);
__device__ void read_result_1cu_2th(volatile uint8_t* __restrict__ output, volatile uint8_t* __restrict__ fim_data,
                                    FimBankType bank_type, int out_dim, uint32_t s_row, uint32_t s_col,
                                    uint64_t offset);

__device__ void read_result_2bank_1cu_2th(volatile uint8_t* __restrict__ output,
                                          volatile uint8_t* __restrict__ fim_data, int out_dim, uint32_t s_row,
                                          uint32_t s_col, uint64_t offset);

__device__ void compute_gemv_2bank_1cu_2th(volatile uint8_t* __restrict__ fim_ctr,
                                           volatile uint8_t* __restrict__ fim_weight,
                                           volatile uint8_t* __restrict__ fim_input, int num_in_tile, int num_out_tile,
                                           int input_tile, int output_tile, int batch_idx, FimBankType bank_type,
                                           uint64_t offset);

__device__ void compute_elt_op_1cu_2th(volatile uint8_t* __restrict__ fim_input0,
                                       volatile uint8_t* __restrict__ fim_input1,
                                       volatile uint8_t* __restrict__ fim_output, int num_tile, uint64_t offset);

__device__ void compute_elt_op_64cu_16th(volatile uint8_t* __restrict__ fim_input0,
                                         volatile uint8_t* __restrict__ fim_input1,
                                         volatile uint8_t* __restrict__ fim_output, int num_tile, uint64_t offset);
/* 64CU 2TH functions */

__device__ void add_transaction_all_64cu_2th(volatile uint8_t* __restrict__ fim_addr, bool is_write, uint32_t bg,
                                             uint32_t bank, uint32_t row, uint32_t col, uint8_t* burst, uint64_t offset,
                                             int loop_cnt = 1);
__device__ void change_fim_mode_64cu_2th(volatile uint8_t* __restrict__ fim_ctr, FimMode mode1, FimMode mode2,
                                         uint8_t* change_mode_bin, uint64_t offset);
__device__ void park_64cu_2th(volatile uint8_t* __restrict__ fim_addr, uint64_t offset);
__device__ void program_crf_64cu_2th(volatile uint8_t* __restrict__ fim_ctr, uint8_t* crf_bin, uint32_t cmd_size,
                                     uint64_t offset);
__device__ void compute_gemv_2bank_64cu_2th(volatile uint8_t* __restrict__ fim_ctr,
                                            volatile uint8_t* __restrict__ fim_weight,
                                            volatile uint8_t* __restrict__ fim_input, int num_in_tile, int num_out_tile,
                                            int input_tile, int output_tile, int batch_idx, FimBankType bank_type,
                                            uint64_t offset);

/* Multi blocks with multi threads functions according to host setting */

__device__ void add_transaction_all(volatile uint8_t* __restrict__ fim_addr, bool is_write, uint32_t bg, uint32_t bank,
                                    uint32_t row, uint32_t col, uint8_t* burst, uint64_t offset, int loop_cnt = 1);
__device__ void change_fim_mode(volatile uint8_t* __restrict__ fim_ctr, FimMode mode1, FimMode mode2,
                                uint8_t* change_mode_bin, uint64_t offset);
__device__ void park(volatile uint8_t* __restrict__ fim_addr, uint64_t offset);
__device__ void program_crf(volatile uint8_t* __restrict__ fim_ctr, uint8_t* crf_bin, uint32_t cmd_size,
                            uint64_t offset);
__device__ void compute_gemv_2bank(volatile uint8_t* __restrict__ fim_ctr, volatile uint8_t* __restrict__ fim_weight,
                                   volatile uint8_t* __restrict__ fim_input, int num_in_tile, int num_out_tile,
                                   int input_tile, int output_tile, int batch_idx, FimBankType bank_type,
                                   uint64_t offset);

#endif /* _FIM_UTIL_H_ */
