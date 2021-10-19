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

#define DIM_OUT_PIM (3200)

__host__ void get_pim_block_info(PimBlockInfo* fbi) { memcpy(fbi, &vega20_fbi, sizeof(PimBlockInfo)); }

size_t get_aligned_size(PimDesc* pim_desc, PimMemFlag mem_flag, PimBo* pim_bo)
{
    size_t size;

    PimBShape bs = pim_desc->bshape;

    if (mem_flag == GEMV_INPUT) {
        bs.h = 1;
    } else if (mem_flag == GEMV_WEIGHT) {
        bs.n = 1;
    } else if (mem_flag == GEMV_WEIGHT_T) {
        bs.n = 1;
        bs.t = true;
    } else if (mem_flag == GEMV_OUTPUT) {
        bs.w = bs.h;
        bs.h = 1;
    } else {
        std::cout << "[Error] " << __FUNCTION__ << ": wrong mem_flag" << std::endl;
        return 0;
    }

    pim_bo->bshape_r = pim_desc->bshape_r;
    pim_bo->bshape = {bs.w, bs.h, bs.c, bs.n, bs.t};

    size = bs.n * bs.c * bs.h * bs.w;

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

bool is_pim_available(PimBo* out, PimBo* op0, PimBo* op1, PimOpType op_type)
{
    bool ret = false;

    switch (op_type) {
        case OP_GEMV:
            ret = is_pim_gemv(op1);
            break;
        default:
            ret = false;
    }

    return ret;
}

bool is_pim_gemv(PimBo* bo)
{
    /* TODO: find optimal shape to execute PIM ops */
    if ((bo->bshape_r.h >= DIM_OUT_PIM || bo->bshape_r.n > 1) && !is_transposed(bo))
        return true;
    else
        return false;
}

bool is_transposed(PimBo* bo) { return bo->bshape.t; }
