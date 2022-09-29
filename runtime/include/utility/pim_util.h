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
__host__ __device__ uint64_t addr_gen_safe(uint32_t chan, uint32_t rank, uint32_t bg, uint32_t bank, uint32_t& row,
                                           uint32_t& col);
void transpose_pimbo(PimBo* dst, PimBo* src);
void set_pimbo_t(PimBo* dst, PimBo* src);
size_t get_aligned_size(PimDesc* pim_desc, PimMemFlag mem_flag, PimBo* pim_bo);
void set_pimbo(PimGemmDesc* pim_gemm_desc, PimMemType mem_type, PimMemFlag mem_flag, PimBo* pim_bo);
void pad_data(void* input, int in_size, int in_nsize, int batch_size, PimMemFlag mem_flag);
void pad_data(void* input, PimDesc* pim_desc, PimMemType mem_type, PimMemFlag mem_flag);
void align_shape(PimDesc* pim_desc, PimOpType op_type);
void align_gemm_shape(PimGemmDesc* pim_gemm_desc);
bool is_pim_applicable(PimBo* wei);
bool is_transposed(PimBo* wei);
bool is_pim_gemv_list_available(PimBo* output, PimBo* vector, PimBo* matrix);
bool check_chwise_gemm_bo(PimBo* bo);
size_t PrecisionSize(const PimBo* bo);

#endif /* _PIM_UTIL_H_ */
