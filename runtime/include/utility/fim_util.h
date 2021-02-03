/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_UTIL_H_
#define _FIM_UTIL_H_

#include "fim_data_types.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "manager/FimInfo.h"
#include "utility/fim_log.h"

/* TODO: get VEGA20 scheme from device driver */
static __constant__ FimBlockInfo vega20_fbi = {
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
#endif

__device__ void R_CMD(volatile uint8_t* __restrict__ addr);
__device__ void W_CMD(volatile uint8_t* __restrict__ addr);
__device__ void W_CMD_R(volatile uint8_t* __restrict__ addr, volatile uint8_t* __restrict__ src);

__device__ void B_CMD(int type);

#endif /* _FIM_UTIL_H_ */
