#ifndef _FIM_UTIL_H_
#define _FIM_UTIL_H_

#include "executor/fim_hip_kernels/fim_crf_bins.h"
#include "fim_data_types.h"
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"

using half_float::half;

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
};

__host__ inline void get_fim_block_info(FimBlockInfo* fbi) { memcpy(fbi, &vega20_fbi, sizeof(FimBlockInfo)); }

__host__ __device__ inline uint32_t mask_by_bit(uint32_t value, uint32_t start, uint32_t end)
{
    int length = start - end + 1;
    value = value >> end;
    return value & ((1 << length) - 1);
}

__host__ __device__ inline uint64_t addr_gen(uint32_t chan, uint32_t rank, uint32_t bankgroup, uint32_t bank,
                                             uint32_t row, uint32_t col)
{
    uint64_t addr = 0;

    FimBlockInfo* fbi = &vega20_fbi;

    addr = rank;

    addr <<= fbi->num_row_bit;
    addr |= row;

    addr <<= fbi->num_col_high_bit;
    addr |= mask_by_bit(col, 4, 2);

    addr <<= fbi->num_bank_high_bit;
    addr |= mask_by_bit(bank, 1, 1);

    addr <<= fbi->num_bankgroup_bit;
    addr |= bankgroup;

    addr <<= fbi->num_bank_low_bit;
    addr |= mask_by_bit(bank, 0, 0);

    addr <<= fbi->num_chan_bit - 1;
    addr |= mask_by_bit(chan, fbi->num_chan_bit - 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(col, 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(chan, 0, 0);

    addr <<= 1;
    addr |= mask_by_bit(col, 0, 0);

    addr <<= fbi->num_offset_bit;

    return addr;
}

__host__ __device__ inline uint64_t addr_gen_safe(uint32_t chan, uint32_t rank, uint32_t bg, uint32_t bank,
                                                  uint32_t& row, uint32_t& col)
{
    FimBlockInfo* fbi = &vega20_fbi;

    while (col >= fbi->num_col / fbi->bl) {
        row++;
        col -= (fbi->num_col / fbi->bl);
    }

    if (row >= fbi->num_row) {
    }

    return addr_gen(chan, rank, bg, bank, row, col);
}

#ifdef EMULATOR
extern uint64_t g_fba;
extern int g_ridx;
extern int g_idx[64];
extern int m_width;
extern FimMemTraceData* g_fmtd16;

__device__ inline void GEN_WRITE_CMD(volatile uint8_t* __restrict__ dst, volatile uint8_t* __restrict__ src)
{
    int bid = hipBlockIdx_x;
    int tid = hipThreadIdx_x;
    int ridx = atomicAdd(&g_ridx, 1);

    g_fmtd16[ridx].block_id = bid;
    g_fmtd16[ridx].thread_id = tid;
    g_fmtd16[ridx].cmd = 'W';
    g_fmtd16[ridx].addr = (uint64_t)dst - g_fba;
    memcpy(g_fmtd16[ridx].data, (uint8_t*)src, 16);
}

__device__ inline void GEN_READ_CMD(volatile uint8_t* __restrict__ dst, volatile uint8_t* __restrict__ src,
                                    bool is_output = false)
{
    int bid = hipBlockIdx_x;
    int tid = hipThreadIdx_x;
    int ridx = atomicAdd(&g_ridx, 1);

    g_fmtd16[ridx].block_id = bid;
    g_fmtd16[ridx].thread_id = tid;
    g_fmtd16[ridx].cmd = (is_output == true) ? 'O' : 'R';
    g_fmtd16[ridx].addr = (uint64_t)src - g_fba;
}

__device__ inline void BLOCK_SYNC(int cu_ch_idx = 0, bool block_all_chan = true)
{
    __syncthreads();
    if (hipThreadIdx_x % 2 == 0) {
        FimBlockInfo* fbi = &vega20_fbi;
        int tid = hipThreadIdx_x;
        int num_chan = fbi->num_fim_chan;
        int ridx;

        if (block_all_chan == true) {
            for (int cidx = 0; cidx < num_chan; cidx++) {
                ridx = atomicAdd(&g_ridx, 1);
                g_fmtd16[ridx].block_id = cidx;
                g_fmtd16[ridx].thread_id = tid;
                g_fmtd16[ridx].cmd = 'B';
                g_fmtd16[ridx].addr = 0;
            }
        } else {
            ridx = atomicAdd(&g_ridx, 1);
            g_fmtd16[ridx].block_id = cu_ch_idx;
            g_fmtd16[ridx].thread_id = tid;
            g_fmtd16[ridx].cmd = 'B';
            g_fmtd16[ridx].addr = 0;
        }
    }
}

#else /* TARGET */

__device__ inline void GEN_WRITE_CMD(volatile uint8_t* __restrict__ dst, volatile uint8_t* __restrict__ src)
{
    asm volatile("global_store_dwordx4 %0, v[27:30], off\n\t" ::"v"(dst) : "v27", "v28", "v29", "v30");
}

__device__ inline void GEN_READ_CMD(volatile uint8_t* __restrict__ dst, volatile uint8_t* __restrict__ src,
                                    bool is_output = false)
{
    asm volatile("global_load_dwordx4 v[27:30], %0, off\n\t" ::"v"(src) : "v27", "v28", "v29", "v30");
}

__device__ inline void BLOCK_SYNC(void) { __syncthreads(); }

#endif /* EMULATOR */

__device__ inline void add_transaction_all_1cu_2th(volatile uint8_t* __restrict__ fim_addr, bool is_write, uint32_t bg,
                                                   uint32_t bank, uint32_t row, uint32_t col, uint8_t* burst,
                                                   uint64_t offset, int loop_cnt = 1)
{
    uint64_t t_addr;
    FimBlockInfo* fbi = &vega20_fbi;

    for (int cidx = 0; cidx < fbi->num_fim_chan; cidx++) {
        for (int rank = 0; rank < fbi->num_fim_rank; rank++) {
            uint32_t local_row = row;
            uint32_t local_col = col;
            for (int lc = 0; lc < loop_cnt; lc++) {
                t_addr = addr_gen_safe(cidx, rank, bg, bank, local_row, local_col);
                if (is_write) {
                    GEN_WRITE_CMD(&fim_addr[t_addr + offset], burst + offset);
                } else {
                    GEN_READ_CMD(null_bst + offset, &fim_addr[t_addr + offset]);
                }
                local_col++;
            }
        }
    }
    BLOCK_SYNC();
}

__device__ inline void change_fim_mode_1cu_2th(volatile uint8_t* __restrict__ fim_ctr, FimMode mode1, FimMode mode2,
                                               uint8_t* change_mode_bin, uint64_t offset)
{
    FimBlockInfo* fbi = &vega20_fbi;

    if (mode1 == SB_MODE) {
        if (mode2 == HAB_MODE) {
            add_transaction_all_1cu_2th(fim_ctr, true, 0, 0, 0x17ff, 0x1f, change_mode_bin, offset);
            add_transaction_all_1cu_2th(fim_ctr, true, 0, 1, 0x17ff, 0x1f, change_mode_bin, offset);
            if (fbi->num_banks >= 2) {
                add_transaction_all_1cu_2th(fim_ctr, true, 2, 0, 0x17ff, 0x1f, change_mode_bin, offset);
                add_transaction_all_1cu_2th(fim_ctr, true, 2, 1, 0x17ff, 0x1f, change_mode_bin, offset);
            }
        }
    } else if (mode1 == HAB_MODE) {
        if (mode2 == SB_MODE) {
            add_transaction_all_1cu_2th(fim_ctr, true, 0, 0, 0x1fff, 0x1f, change_mode_bin, offset);
            add_transaction_all_1cu_2th(fim_ctr, true, 0, 1, 0x1fff, 0x1f, change_mode_bin, offset);
        } else if (mode2 == HAB_FIM_MODE) {
            add_transaction_all_1cu_2th(fim_ctr, true, 0, 0, 0x3fff, 0x0, change_mode_bin, offset);
        }
    } else if (mode1 == HAB_FIM_MODE) {
        if (mode2 == HAB_MODE) {
            add_transaction_all_1cu_2th(fim_ctr, true, 0, 0, 0x3fff, 0x0, change_mode_bin, offset);
        }
    }
    BLOCK_SYNC();
}

__device__ inline void park_in_1cu_2th(volatile uint8_t* __restrict__ fim_ctr, uint64_t offset)
{
    uint32_t cidx;
    uint32_t rank;
    uint32_t b;
    uint32_t bg;
    uint64_t t_addr;

    FimBlockInfo* fbi = &vega20_fbi;

    BLOCK_SYNC();
    for (cidx = 0; cidx < fbi->num_fim_chan; cidx++) {
        for (rank = 0; rank < fbi->num_fim_rank; rank++) {
            for (b = 0; b < fbi->num_banks / fbi->num_bank_groups; b++) {
                for (bg = 0; bg < fbi->num_bank_groups; bg++) {
                    t_addr = addr_gen(cidx, rank, bg, b, (1 << 12), 0);
                    GEN_READ_CMD(null_bst + offset, &fim_ctr[t_addr + offset]);
                }
            }
        }
    }
    BLOCK_SYNC();
}

__device__ inline void park_out_1cu_2th(volatile uint8_t* __restrict__ fim_ctr, uint64_t offset)
{
    uint32_t cidx;
    uint32_t rank;
    uint64_t t_addr;
    FimBlockInfo* fbi = &vega20_fbi;

    for (cidx = 0; cidx < fbi->num_fim_chan; cidx++) {
        for (rank = 0; rank < fbi->num_fim_rank; rank++) {
            t_addr = addr_gen(cidx, rank, 0, 0, (1 << 12), 0);
            GEN_READ_CMD(null_bst + offset, &fim_ctr[t_addr + offset]);

            t_addr = addr_gen(cidx, rank, 0, 1, (1 << 12), 0);
            GEN_READ_CMD(null_bst + offset, &fim_ctr[t_addr + offset]);
        }
    }
    BLOCK_SYNC();
}

__device__ inline void program_crf_1cu_2th(volatile uint8_t* __restrict__ fim_ctr, uint8_t* crf_bin, uint32_t cmd_size,
                                           uint64_t offset)
{
    FimBlockInfo* fbi = &vega20_fbi;

    for (int i = 0; i < cmd_size; i += fbi->trans_size) {
        add_transaction_all_1cu_2th(fim_ctr, true, 0, 1, 0x3fff, 0x4 + i, crf_bin + i, offset);
    }
}

__device__ inline void compute_elt_add_1cu_2th(volatile uint8_t* __restrict__ fim_data, int num_tile, uint64_t offset)
{
    FimBlockInfo* fbi = &vega20_fbi;

    for (int i = 0; i < num_tile; i++) {
        add_transaction_all_1cu_2th(fim_data, false, 0, 0, 0, fbi->num_grf * i, null_bst, offset, fbi->num_grf);
        add_transaction_all_1cu_2th(fim_data, false, 0, 1, 0, fbi->num_grf * i, null_bst, offset, fbi->num_grf);
        add_transaction_all_1cu_2th(fim_data, true, 0, 1, 0, fbi->num_grf * (num_tile + i), null_bst, offset,
                                    fbi->num_grf);
    }
}

__device__ inline int get_num_tile(int dim)
{
    FimBlockInfo* fbi = &vega20_fbi;
    int num_parallelism = fbi->num_fim_blocks * fbi->num_fim_chan * fbi->num_fim_rank * fbi->num_grf;
    int tile = dim / num_parallelism;

    return tile;
}

__device__ inline int get_result_col(int dim)
{
    FimBlockInfo* fbi = &vega20_fbi;

    return dim / (fbi->num_fim_blocks * fbi->num_fim_chan * fbi->num_fim_rank);
}

__device__ inline int gemv_get_result_col(int input_dim, int output_dim, int num_in_tile, int num_out_tile)
{
    FimBlockInfo* fbi = &vega20_fbi;

    return num_out_tile * num_in_tile / 2 * fbi->num_grf_A * fbi->num_grf_B;
}

__device__ inline void read_result_1cu_2th(volatile uint8_t* __restrict__ output,
                                           volatile uint8_t* __restrict__ fim_data, FimBankType bank_type, int out_dim,
                                           uint32_t s_row, uint32_t s_col, uint64_t offset)
{
    FimBlockInfo* fbi = &vega20_fbi;
    uint32_t cidx = 0;
    uint32_t rank = 0;
    uint32_t bg = 0;
    uint32_t bank = 0;
    uint32_t row = 0;
    uint32_t col = 0;
    uint64_t t_addr;

    for (int x = 0; x < out_dim; x += fbi->num_grf) {
        row = s_row;
        col = s_col;
        for (int grf_idx = 0; grf_idx < fbi->num_grf; grf_idx++) {
            t_addr = addr_gen_safe(cidx, rank, bg, bank + (uint32_t)bank_type, row, col);
            GEN_READ_CMD(output + x + grf_idx + offset, &fim_data[t_addr + offset], true);
            col++;
        }

        bank += (fbi->num_banks / fbi->num_fim_blocks);
        if (bank >= (fbi->num_banks / fbi->num_bank_groups)) {
            bg++;
            bank = 0;
        }
        if (bg >= fbi->num_bank_groups) {
            bg = 0;
            rank++;
        }
        if (rank >= fbi->num_fim_rank) {
            rank = 0;
            cidx++;
        }
        if (cidx >= fbi->num_fim_chan) {
            cidx = 0;
            s_row = row;
            s_col = col;
        }
    }
}

__device__ inline void compute_gemv_2bank_1cu_2th(volatile uint8_t* __restrict__ fim_ctr,
                                                  volatile uint8_t* __restrict__ fim_weight,
                                                  volatile uint8_t* __restrict__ fim_input, int num_in_tile,
                                                  int num_out_tile, int input_tile, int output_tile,
                                                  FimBankType bank_type, uint64_t offset)
{
    FimBlockInfo* fbi = &vega20_fbi;
    uint64_t addr;
    uint32_t row = 0;
    uint32_t col = (fbi->num_grf_A * fbi->num_grf_B) * (input_tile / 2 + output_tile * num_in_tile);

    for (int cidx = 0; cidx < fbi->num_fim_chan; cidx++) {
        for (int rank = 0; rank < fbi->num_fim_rank; rank++) {
            for (int gidx = 0; gidx < fbi->num_grf_A; gidx++) {
                addr = addr_gen(cidx, rank, 0, 1, 0x3fff, 0x8 + gidx);
                GEN_WRITE_CMD(&fim_ctr[addr + offset], fim_input + input_tile * fbi->num_grf_A + gidx + offset);
            }
            BLOCK_SYNC(cidx, false);
        }
    }
    add_transaction_all_1cu_2th(fim_ctr, false, 0, (int)bank_type, row, col, null_bst, offset, fbi->num_grf_A * fbi->num_grf_B);
}

#ifdef EMULATOR
__device__ inline void R_CMD(uint8_t* addr)
{
    int row = hipBlockIdx_x * m_width;
    int midx = row + atomicAdd(&g_idx[hipBlockIdx_x], 1);

    memset(g_fmtd16[midx].data, 0, 16);
    g_fmtd16[midx].block_id = hipBlockIdx_x;
    g_fmtd16[midx].thread_id = hipThreadIdx_x;
    g_fmtd16[midx].addr = (uint64_t)addr - g_fba;
    g_fmtd16[midx].cmd = 'R';
}

__device__ inline void W_CMD(uint8_t* addr)
{
    int row = hipBlockIdx_x * m_width;
    int midx = row + atomicAdd(&g_idx[hipBlockIdx_x], 1);

    memset(g_fmtd16[midx].data, 0, 16);
    g_fmtd16[midx].block_id = hipBlockIdx_x;
    g_fmtd16[midx].thread_id = hipThreadIdx_x;
    g_fmtd16[midx].addr = (uint64_t)addr - g_fba;
    g_fmtd16[midx].cmd = 'W';
}

__device__ inline void W_CMD_R(uint8_t* addr, uint8_t* src)
{
    int row = hipBlockIdx_x * m_width;
    int midx = row + atomicAdd(&g_idx[hipBlockIdx_x], 1);

    memcpy(g_fmtd16[midx].data, (uint8_t*)src, 16);
    g_fmtd16[midx].block_id = hipBlockIdx_x;
    g_fmtd16[midx].thread_id = hipThreadIdx_x;
    g_fmtd16[midx].addr = (uint64_t)addr - g_fba;
    g_fmtd16[midx].cmd = 'W';
}

__device__ inline void B_CMD(int type)
{
    int row = hipBlockIdx_x * m_width;
    int midx = row + atomicAdd(&g_idx[hipBlockIdx_x], 1);

    memset(g_fmtd16[midx].data, 0, 16);
    g_fmtd16[midx].block_id = hipBlockIdx_x;
    g_fmtd16[midx].thread_id = hipThreadIdx_x;
    g_fmtd16[midx].addr = 0;
    g_fmtd16[midx].cmd = 'B';

    if (type == 0)
        __syncthreads();
    else
        __threadfence();
}

__device__ inline void record(int bid, char mtype, uint64_t paddr)
{
    int row = bid * m_width;
    int midx = row + g_idx[bid]++;

    memset(g_fmtd16[midx].data, 0, 16);
    g_fmtd16[midx].block_id = bid;
    g_fmtd16[midx].thread_id = 0;
    g_fmtd16[midx].addr = paddr;
    g_fmtd16[midx].cmd = mtype;
}
#else
__device__ inline void R_CMD(uint8_t* addr)
{
    asm volatile("global_load_dwordx4 v[24:27], %0, off\n\t" ::"v"(addr) : "v24", "v25", "v26", "v37");
}

__device__ inline void W_CMD(uint8_t* addr)
{
    asm volatile("global_store_dwordx4 %0, v[24:27], off\n\t" ::"v"(addr) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD_R(uint8_t* addr, uint8_t* src)
{
    if (hipThreadIdx_x == 0) {
        asm volatile("global_load_dwordx4 v[20:23], %0, off\n\t" ::"v"(src) : "v20", "v21", "v22", "v23");
        asm volatile("global_store_dwordx4 %0, v[20:23], off\n\t" ::"v"(addr) : "v20", "v21", "v22", "v23");
    } else {
        asm volatile("global_load_dwordx4 v[24:27], %0, off\n\t" ::"v"(src) : "v24", "v25", "v26", "v37");
        asm volatile("global_store_dwordx4 %0, v[24:27], off\n\t" ::"v"(addr) : "v24", "v25", "v26", "v27");
    }
}

__device__ inline void B_CMD(int type)
{
    if (type == 0)
        __syncthreads();
    else
        __threadfence();
}
#endif
#endif /* _FIM_UTIL_H_ */
