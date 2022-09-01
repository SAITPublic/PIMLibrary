/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_INFO_H_
#define _PIM_INFO_H_

#include <stdint.h>

#define DIM_OUT_PIM (3200)
#define PIM_GEMV_IN_ALIGN (256)
#define PIM_GEMV_OUT_ALIGN (4096)
#define PIM_ELTWISE_ALIGN (256 * 1024)

typedef enum __PimAddrMap {
    AMDGPU_VEGA20,
} PimAddrMap;

typedef struct __PimBlockInfo {
    PimAddrMap pim_addr_map;
    int num_banks;
    int num_bank_groups;
    int num_rank_bit;
    int num_row_bit;
    int num_col_high_bit;
    int num_bank_high_bit;
    int num_bankgroup_bit;
    int num_bank_low_bit;
    int num_chan_bit;
    int num_col_low_bit;
    int num_offset_bit;
    int num_grf;
    int num_grf_A;
    int num_grf_B;
    int num_srf;
    int num_col;
    int num_row;
    int bl;
    int num_pim_blocks;
    int num_pim_rank;
    int num_pim_chan;
    int trans_size;
    int num_out_per_grf;
} PimBlockInfo;

static PimBlockInfo vega20_pbi = {
    .pim_addr_map = AMDGPU_VEGA20,
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
    .num_pim_blocks = 8,
    .num_pim_rank = 1,
    .num_pim_chan = 64,
    .trans_size = 32,
    .num_out_per_grf = 16,
};

typedef struct __PimMemTraceData {
    uint8_t data[32];
    uint64_t addr;
    int block_id;
    int thread_id;
    char cmd;
} PimMemTraceData;

typedef enum __PimBankType {
    EVEN_BANK,
    ODD_BANK,
    ALL_BANK,
} PimBankType;

typedef enum __PimMode {
    SB_MODE,
    HAB_MODE,
    HAB_PIM_MODE,
} PimMode;

typedef enum __PimGemvType {
    TILE_ACCUM,
    TILE_TREE,
    NEXT_PIM,
} PimGemvType;

typedef enum __PimKrnlType {
    OPTIMAL,
    PIM,
    CUSTOM_GPU,
} PimKrnlType;

#ifdef EMULATOR
typedef struct __PimMemTracer {
    uint64_t g_fba;
    PimMemTraceData* g_fmtd16;
    int g_ridx[64];
    int g_idx[64];
    int m_width;
} PimMemTracer;
#endif

#endif /* _PIM_INFO_H_ */
