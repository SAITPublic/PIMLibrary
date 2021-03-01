/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "utility/fim_util.h"

__host__ void get_fim_block_info(FimBlockInfo* fbi) { memcpy(fbi, &vega20_fbi, sizeof(FimBlockInfo)); }

size_t get_aligned_size(FimDesc* fim_desc, FimMemFlag mem_flag, FimBo* fim_bo)
{
    size_t size;

    int n = fim_desc->bshape.n;
    int c = fim_desc->bshape.c;
    int h = fim_desc->bshape.h;
    int w = fim_desc->bshape.w;

    if (mem_flag == GEMV_INPUT) {
        h = 1;
    } else if (mem_flag == GEMV_WEIGHT) {
        n = 1;
    } else if (mem_flag == GEMV_OUTPUT) {
        w = 1;
    }
    fim_bo->bshape_r = fim_desc->bshape_r;
    fim_bo->bshape = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};

    size = n * c * h * w;

    return size;
}

void align_shape(FimDesc* fim_desc, FimOpType op_type)
{
    FimBShape bs = fim_desc->bshape_r;

    if (op_type == OP_GEMV) {
        bs.w = 256 * ceil((float)bs.w / 256);
        bs.h = 4096 * ceil((float)bs.h / 4096);
    } else {
        bs.w = (256 * 1024) * ceil((float)bs.w / (256 * 1024));
    }
    fim_desc->bshape = bs;
}

void pad_data(void* input, FimDesc* fim_desc, FimMemType mem_type, FimMemFlag mem_flag)
{
    if (mem_flag == GEMV_INPUT && mem_type == MEM_TYPE_HOST) {
        if (mem_type == MEM_TYPE_HOST) {
            for (int i = 0; i < fim_desc->bshape.n; i++) {
                for (int j = fim_desc->bshape_r.w; j < fim_desc->bshape.w; j++) {
                    ((half*)input)[i * fim_desc->bshape.w + j] = half(0);
                }
            }
        } else {
            if (fim_desc->bshape_r.w != fim_desc->bshape.w)
                hipMemset(input, 0, fim_desc->bshape.n * fim_desc->bshape.w * sizeof(half));
        }
    }

    if (mem_flag == ELT_OP) {
        int padded_size = fim_desc->bshape.n * fim_desc->bshape.c * fim_desc->bshape.h * fim_desc->bshape.w;
        int real_size = fim_desc->bshape_r.n * fim_desc->bshape_r.c * fim_desc->bshape_r.h * fim_desc->bshape_r.w;
        for (int i = real_size; i < padded_size; i++) {
            ((half*)input)[i] = half(0);
        }
    }
}

void pad_data(void* input, int in_size, int in_nsize, int batch_size, FimMemFlag mem_flag)
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
