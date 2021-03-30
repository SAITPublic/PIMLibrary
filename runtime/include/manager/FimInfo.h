/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_INFO_H_
#define _FIM_INFO_H_

typedef enum __FimAddrMap {
    AMDGPU_VEGA20,
} FimAddrMap;

typedef struct __FimBlockInfo {
    FimAddrMap fim_addr_map;
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
    int num_fim_blocks;
    int num_fim_rank;
    int num_fim_chan;
    int trans_size;
    int num_out_per_grf;
} FimBlockInfo;

typedef struct __FimMemTraceData {
    uint8_t data[32];
    uint64_t addr;
    int block_id;
    int thread_id;
    char cmd;
} FimMemTraceData;

typedef enum __FimBankType {
    EVEN_BANK,
    ODD_BANK,
    ALL_BANK,
} FimBankType;

typedef enum __FimMode {
    SB_MODE,
    HAB_MODE,
    HAB_FIM_MODE,
} FimMode;

#endif /* _FIM_INFO_H_ */
