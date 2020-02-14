#ifndef _FIM_UTIL_H_
#define _FIM_UTIL_H_

#include "executor/fim_hip_kernels/fim_crf_bins.h"
#include "fim_data_types.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"

/* TODO: get VEGA20 scheme from device driver */
FimBlockInfo vega20_fbi = {
    .fim_addr_map = AMDGPU_VEGA20,
    .num_banks = 16,
    .num_bank_groups = 8,
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
    .num_col = 128,
    .num_row = 16384,
    .bl = 4,
    .num_fim_blocks = 8,
    .num_fim_rank = 1,
    .num_fim_chan = 64,
};

uint8_t null_bst[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

__host__ inline int get_fim_block_info(FimBlockInfo* fbi) { memcpy(fbi, &vega20_fbi, sizeof(FimBlockInfo)); }

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

__device__ inline void GEN_WRITE_CMD(volatile uint8_t* __restrict__ dst, volatile uint8_t* __restrict__ src)
{
#ifdef EMULATOR_PATH

#else
    asm volatile("global_store_dwordx4 %0, v[27:30], off\n\t" ::"v"(dst) : "v27", "v28", "v29", "v30");
#endif
}

__device__ inline void GEN_READ_CMD(volatile uint8_t* __restrict__ dst, volatile uint8_t* __restrict__ src)
{
#ifdef EMULATOR_PATH

#else
    asm volatile("global_load_dwordx4 v[27:30], %0, off\n\t" ::"v"(src) : "v27", "v28", "v29", "v30");
#endif
}

__device__ inline void add_transaction_all(volatile uint8_t* __restrict__ fim_addr, bool is_write, uint32_t bg,
                                           uint32_t bank, uint32_t row, uint32_t col, uint8_t* burst, int loop_cnt = 1)
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
                    GEN_WRITE_CMD(&fim_addr[t_addr], null_bst);
                    GEN_WRITE_CMD(&fim_addr[t_addr], null_bst + 0x10);
                } else {
                    GEN_READ_CMD(null_bst, &fim_addr[t_addr]);
                    GEN_READ_CMD(null_bst + 0x10, &fim_addr[t_addr]);
                }
                local_col++;
            }
        }
    }
    __syncthreads();
}

__device__ inline void change_fim_mode(volatile uint8_t* __restrict__ fim_ctr, FimMode mode1, FimMode mode2)
{
    FimBlockInfo* fbi = &vega20_fbi;

    if (mode1 == SB_MODE) {
        if (mode2 == HAB_MODE) {
            add_transaction_all(fim_ctr, true, 0, 0, 0x17ff, 0x1f, null_bst);
            add_transaction_all(fim_ctr, true, 0, 1, 0x17ff, 0x1f, null_bst);
            if (fbi->num_banks >= 2) {
                add_transaction_all(fim_ctr, true, 2, 0, 0x17ff, 0x1f, null_bst);
                add_transaction_all(fim_ctr, true, 2, 1, 0x17ff, 0x1f, null_bst);
            }
        }
    } else if (mode1 == HAB_MODE) {
        if (mode2 == SB_MODE) {
            add_transaction_all(fim_ctr, true, 0, 0, 0x1fff, 0x1f, null_bst);
            add_transaction_all(fim_ctr, true, 0, 1, 0x1fff, 0x1f, null_bst);
        } else if (mode2 == HAB_FIM_MODE) {
            add_transaction_all(fim_ctr, true, 0, 0, 0x3fff, 0x0, hab_to_hab_fim);
        }
    } else if (mode1 == HAB_FIM_MODE) {
        if (mode2 == HAB_MODE) {
            add_transaction_all(fim_ctr, true, 0, 0, 0x3fff, 0x0, hab_fim_to_hab);
        }
    }
    __syncthreads();
}

__device__ inline void park_in(volatile uint8_t* __restrict__ fim_ctr)
{
    uint32_t cidx;
    uint32_t rank;
    uint32_t b;
    uint32_t bg;
    uint64_t t_addr;

    FimBlockInfo* fbi = &vega20_fbi;

    __syncthreads();
    for (cidx = 0; cidx < fbi->num_fim_chan; cidx++) {
        for (rank = 0; rank < fbi->num_fim_rank; rank++) {
            for (b = 0; b < fbi->num_banks / fbi->num_bank_groups; b++) {
                for (bg = 0; bg < fbi->num_bank_groups; bg++) {
                    t_addr = addr_gen(cidx, rank, bg, b, (1 << 12), 0);
                    GEN_READ_CMD(null_bst, &fim_ctr[t_addr]);
                    GEN_READ_CMD(null_bst, &fim_ctr[t_addr + 0x10]);
                }
            }
        }
    }
    __syncthreads();
}

__device__ inline void park_out(volatile uint8_t* __restrict__ fim_ctr)
{
    uint32_t cidx;
    uint32_t rank;
    uint64_t t_addr;
    FimBlockInfo* fbi = &vega20_fbi;

    for (cidx = 0; cidx < fbi->num_fim_chan; cidx++) {
        for (rank = 0; rank < fbi->num_fim_rank; rank++) {
            t_addr = addr_gen(cidx, rank, 0, 0, (1 << 12), 0);
            GEN_READ_CMD(null_bst, &fim_ctr[t_addr]);
            GEN_READ_CMD(null_bst, &fim_ctr[t_addr + 0x10]);

            t_addr = addr_gen(cidx, rank, 0, 1, (1 << 12), 0);
            GEN_READ_CMD(null_bst, &fim_ctr[t_addr]);
            GEN_READ_CMD(null_bst, &fim_ctr[t_addr + 0x10]);
        }
    }
    __syncthreads();
}

__device__ inline void program_crf(volatile uint8_t* __restrict__ fim_ctr, uint8_t* elt_add_crf, uint32_t cmd_size)
{
    int i;
    int offset = 0;

    for (i = 0; i < 4; i++) {
        if (i * 8 >= cmd_size) break;
        add_transaction_all(fim_ctr, true, 0, 1, 0x3fff, 0x4 + i, elt_add_crf + offset);
        offset += 32;
    }
}

__device__ inline void compute_elt_add(volatile uint8_t* __restrict__ fim_data, int num_tile)
{
    FimBlockInfo* fbi = &vega20_fbi;

    for (int i = 0; i < num_tile; i++) {
        add_transaction_all(fim_data, false, 0, 0, 0, fbi->num_grf * i, null_bst, fbi->num_grf);
        add_transaction_all(fim_data, false, 0, 1, 0, fbi->num_grf * i, null_bst, fbi->num_grf);
        add_transaction_all(fim_data, true, 0, 1, 0, fbi->num_grf * (num_tile + i), null_bst, fbi->num_grf);
    }
}

__device__ inline int get_num_tile(int dim)
{
    FimBlockInfo* fbi = &vega20_fbi;
    int num_parallelism = fbi->num_fim_blocks * fbi->num_fim_chan * fbi->num_fim_rank * fbi->num_grf;
    int tile = dim / num_parallelism;

    return tile;
}

#endif /* _FIM_UTIL_H_ */
