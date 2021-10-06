/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "utility/pim_util.h"

__host__ void get_pim_block_info(PimBlockInfo* fbi) { memcpy(fbi, &vega20_fbi, sizeof(PimBlockInfo)); }

size_t get_aligned_size(PimDesc* pim_desc, PimMemFlag mem_flag, PimBo* pim_bo)
{
    size_t size;

    int n = pim_desc->bshape.n;
    int c = pim_desc->bshape.c;
    int h = pim_desc->bshape.h;
    int w = pim_desc->bshape.w;

    if (mem_flag == GEMV_INPUT) {
        h = 1;
    } else if (mem_flag == GEMV_WEIGHT) {
        n = 1;
    } else if (mem_flag == GEMV_OUTPUT) {
        w = 1;
    }
    pim_bo->bshape_r = pim_desc->bshape_r;
    pim_bo->bshape = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};

    size = n * c * h * w;

    return size;
}

void align_shape(PimDesc* pim_desc, PimOpType op_type)
{
    PimBShape bs = pim_desc->bshape_r;

    if (op_type == OP_GEMV) {
        bs.w = 256 * ceil((float)bs.w / 256);
        bs.h = 4096 * ceil((float)bs.h / 4096);
    } else {
        bs.w = (256 * 1024) * ceil((float)bs.w / (256 * 1024));
    }
    pim_desc->bshape = bs;
}

void pad_data(void* input, PimDesc* pim_desc, PimMemType mem_type, PimMemFlag mem_flag)
{
    if (mem_flag == GEMV_INPUT && mem_type == MEM_TYPE_HOST) {
        if (mem_type == MEM_TYPE_HOST) {
            for (int i = 0; i < pim_desc->bshape.n; i++) {
                for (int j = pim_desc->bshape_r.w; j < pim_desc->bshape.w; j++) {
                    ((half*)input)[i * pim_desc->bshape.w + j] = half(0);
                }
            }
        } else {
            if (pim_desc->bshape_r.w != pim_desc->bshape.w)
                hipMemset(input, 0, pim_desc->bshape.n * pim_desc->bshape.w * sizeof(half));
        }
    }

    if (mem_flag == ELT_OP) {
        int padded_size = pim_desc->bshape.n * pim_desc->bshape.c * pim_desc->bshape.h * pim_desc->bshape.w;
        int real_size = pim_desc->bshape_r.n * pim_desc->bshape_r.c * pim_desc->bshape_r.h * pim_desc->bshape_r.w;
        for (int i = real_size; i < padded_size; i++) {
            ((half*)input)[i] = half(0);
        }
    }
}

void pad_data(void* input, int in_size, int in_nsize, int batch_size, PimMemFlag mem_flag)
{
    if (mem_flag == GEMV_INPUT) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = in_size; j < in_nsize; j++) {
                ((half*)input)[in_nsize * i + j] = half(0);
            }
        }
    } else {
        for (int i = in_size; i < in_nsize; i++) {
            ((half*)input)[i] = half(0);
        }
    }
}

bool is_pim_gemv(PimBo* bo)
{
    /* TODO: find optimal shape to execute PIM ops */
    if (bo->bshape_r.h >= 3200 || bo->bshape_r.n > 1)
        return true;
    else
        return false;
}
