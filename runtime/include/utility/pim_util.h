/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_UTIL_H_
#define _PIM_UTIL_H_

#include <iostream>
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "manager/PimInfo.h"
#include "pim_data_types.h"
#include "utility/pim_log.h"

__host__ void get_pim_block_info(PimBlockInfo* pbi);

__host__ __device__ inline uint32_t mask_by_bit(uint32_t value, uint32_t start, uint32_t end)
{
    int length = start - end + 1;
    value >>= end;
    return value & ((1 << length) - 1);
}

__host__ __device__ inline uint64_t addr_gen(uint32_t chan, uint32_t rank, uint32_t bankgroup, uint32_t bank, uint32_t row,
                                      uint32_t col)
{
    const PimBlockInfo& pbi = vega20_pbi;
    int num_row_bit = pbi.num_row_bit;
    int num_col_high_bit = pbi.num_col_high_bit;
    int num_bank_high_bit = pbi.num_bank_high_bit;
    int num_bankgroup_bit = pbi.num_bankgroup_bit;
    int num_bank_low_bit = pbi.num_bank_low_bit;
    int num_chan_bit = pbi.num_chan_bit;
    int num_offset_bit = pbi.num_offset_bit;

    uint64_t addr = 0;

    addr = rank;

    addr <<= num_row_bit;
    addr |= row;

    addr <<= num_col_high_bit;
    addr |= mask_by_bit(col, 4, 2);

    addr <<= num_bank_high_bit;
    addr |= mask_by_bit(bank, 1, 1);

    addr <<= num_bankgroup_bit;
    addr |= bankgroup;

    addr <<= num_bank_low_bit;
    addr |= mask_by_bit(bank, 0, 0);

    addr <<= num_chan_bit - 1;
    addr |= mask_by_bit(chan, num_chan_bit - 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(col, 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(chan, 0, 0);

    addr <<= 1;
    addr |= mask_by_bit(col, 0, 0);

    addr <<= num_offset_bit;

#if TARGET && RADEON7
    /* we assume pim kernel run on vega20(32GB) system */
    /* but SAIT server is vega20(16GB) system */
    /* so upper 2bit should be set as 0 for normal work */
    uint64_t mask = 0x1FFFFFFFF;
    addr &= mask;
#endif

    return addr;
}

__host__ __device__ inline uint64_t addr_gen_safe(uint32_t chan, uint32_t rank, uint32_t bg, uint32_t bank, uint32_t& row,
                                           uint32_t& col)
{
    const PimBlockInfo& pbi = vega20_pbi;

    while (col >= pbi.num_col / pbi.bl) {
        row++;
        col -= (pbi.num_col / pbi.bl);
    }

    if (row >= pbi.num_row) {
    }

    return addr_gen(chan, rank, bg, bank, row, col);
}
void transpose_pimbo(PimBo* dst, PimBo* src);
void set_pimbo_t(PimBo* bo0, PimBo* bo1, PimBo* bo2, PimBo* bo3);
void set_pimbo_t(PimBo* inout);
void set_pimbo_t(PimBo* dst, PimBo* src);
size_t get_aligned_size(PimDesc* pim_desc, PimMemFlag mem_flag, PimBo* pim_bo);
void set_pimbo(PimGemmDesc* pim_gemm_desc, PimMemType mem_type, PimMemFlag mem_flag, bool transposed, PimBo* pim_bo);
void pad_data(void* input, int in_size, int in_nsize, int batch_size, PimMemFlag mem_flag);
void pad_data(void* input, PimDesc* pim_desc, PimMemType mem_type, PimMemFlag mem_flag);
void align_shape(PimDesc* pim_desc, PimOpType op_type);
void align_gemm_shape(PimGemmDesc* pim_gemm_desc);
bool is_pim_applicable(PimBo* wei, PimGemmOrder gemm_order);
bool is_pim_gemv_list_available(PimBo* output, PimBo* vector, PimBo* matrix);
bool check_chwise_gemm_bo(PimBo* bo, PimGemmOrder gemm_order);
size_t PrecisionSize(const PimBo* bo);

#endif /* _PIM_UTIL_H_ */
