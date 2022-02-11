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

/* TODO: get VEGA20 scheme from device driver */
static PimBlockInfo vega20_fbi = {
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

__host__ void get_pim_block_info(PimBlockInfo* fbi);
__host__ __device__ uint64_t addr_gen_safe(uint32_t chan, uint32_t rank, uint32_t bg, uint32_t bank, uint32_t& row,
                                           uint32_t& col);
size_t get_aligned_size(PimDesc* pim_desc, PimMemFlag mem_flag, PimBo* pim_bo);
void pad_data(void* input, int in_size, int in_nsize, int batch_size, PimMemFlag mem_flag);
void pad_data(void* input, PimDesc* pim_desc, PimMemType mem_type, PimMemFlag mem_flag);
void align_shape(PimDesc* pim_desc, PimOpType op_type);
bool is_pim_available(PimBo* out, PimBo* op0, PimBo* op1, PimOpType op_type);
bool is_pim_gemv_available(PimBo* bo);
bool is_pim_gemv_list_available(PimBo* output, PimBo* vector, PimBo* matrix);
bool is_transposed(PimBo* bo);
#endif /* _PIM_UTIL_H_ */
