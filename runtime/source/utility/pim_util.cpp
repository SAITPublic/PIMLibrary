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
#include "half.hpp"

__host__ void get_pim_block_info(PimBlockInfo* pbi) { memcpy(pbi, &vega20_pbi, sizeof(PimBlockInfo)); }

void set_pimbo_t(PimBo* dst, PimBo* src)
{
    memcpy(dst, src, sizeof(PimBo));
    dst->bshape.w = src->bshape.h;
    dst->bshape.h = src->bshape.w;
    dst->bshape_r.w = src->bshape_r.h;
    dst->bshape_r.h = src->bshape_r.w;
}

void transpose_pimbo(PimBo* dst, PimBo* src)
{
    int batch = src->bshape.n;
    int channel = src->bshape.c;
    int row = src->bshape.h;
    int col = src->bshape.w;
    half_float::half* in = reinterpret_cast<half_float::half*>(src->data);
    half_float::half* out = reinterpret_cast<half_float::half*>(dst->data);

    for (int iter = 0; iter < batch * channel; iter++) {
        for (int ri = 0; ri < row; ri++) {
            for (int ci = 0; ci < col; ci++) {
                out[ci * row + ri] = in[ri * col + ci];
            }
        }
        in += row * col;
        out += row * col;
    }
}

size_t get_aligned_size(PimDesc* pim_desc, PimMemFlag mem_flag, PimBo* pim_bo)
{
    size_t size;

    PimBShape bs = pim_desc->bshape;
    PimBShape bs_r = pim_desc->bshape_r;

    switch (mem_flag) {
        case ELT_OP:
            break;
        case GEMV_INPUT:
            bs.c = 1;
            bs.w = bs.h;
            bs.h = 1;
            bs_r.c = 1;
            bs_r.w = bs_r.h;
            bs_r.h = 1;
            break;
        case GEMV_WEIGHT:
            bs.n = 1;
            bs.c = 1;
            bs_r.n = 1;
            bs_r.c = 1;
            break;
        case GEMV_OUTPUT:
            bs.c = 1;
            bs.h = 1;
            bs_r.c = 1;
            bs_r.h = 1;
            break;
        default:
            std::cout << "[Error] " << __FUNCTION__ << ": wrong mem_flag" << std::endl;
            return 0;
    }

    pim_bo->bshape_r = {bs_r.n, bs_r.c, bs_r.h, bs_r.w};
    pim_bo->bshape = {bs.n, bs.c, bs.h, bs.w};

    size = (size_t)bs.n * (size_t)bs.c * (size_t)bs.h * (size_t)bs.w;

    return size;
}

void set_pimbo(PimGemmDesc* pim_gemm_desc, PimMemType mem_type, PimMemFlag mem_flag, PimBo* pim_bo)
{
    size_t size = 0;

    PimBShape* bshape = nullptr;
    PimBShape* bshape_r = nullptr;

    switch (mem_flag) {
        case GEMM_INPUT:
            bshape = &pim_gemm_desc->in_bshape;
            bshape_r = &pim_gemm_desc->in_bshape_r;
            break;
        case GEMM_WEIGHT:
            bshape = &pim_gemm_desc->wei_bshape;
            bshape_r = &pim_gemm_desc->wei_bshape_r;
            break;
        case GEMM_OUTPUT:
            bshape = &pim_gemm_desc->out_bshape;
            bshape_r = &pim_gemm_desc->out_bshape_r;
            break;
        case GEMM_BIAS:
            bshape = &pim_gemm_desc->bias_bshape;
            bshape_r = &pim_gemm_desc->bias_bshape_r;
            break;
        default:
            printf("fail to get aligned buffer size");
            return;
    }

    size = bshape->n * bshape->c * bshape->h * bshape->w;
    if (pim_gemm_desc->precision == PIM_FP16) size *= 2;

    pim_bo->size = size;
    pim_bo->mem_type = mem_type;
    pim_bo->precision = pim_gemm_desc->precision;
    pim_bo->bshape = *bshape;
    pim_bo->bshape_r = *bshape_r;
}

void align_gemm_shape(PimGemmDesc* pim_gemm_desc)
{
    int n = pim_gemm_desc->in_bshape_r.n;
    int c = pim_gemm_desc->in_bshape_r.c;
    int in_w = pim_gemm_desc->in_bshape_r.w;
    int out_w = pim_gemm_desc->out_bshape_r.w;
    int aligned_in_w = PIM_GEMV_IN_ALIGN * ceil((float)in_w / PIM_GEMV_IN_ALIGN);
    int aligned_out_w = 0;
    int out_align = n * c * out_w;

    if (out_align % PIM_GEMV_OUT_ALIGN == 0)
        aligned_out_w = out_w;
    else
        aligned_out_w = PIM_GEMV_OUT_ALIGN * ceil((float)out_w / PIM_GEMV_OUT_ALIGN);

    pim_gemm_desc->in_bshape = pim_gemm_desc->in_bshape_r;
    pim_gemm_desc->wei_bshape = pim_gemm_desc->wei_bshape_r;
    pim_gemm_desc->bias_bshape = pim_gemm_desc->bias_bshape_r;
    pim_gemm_desc->out_bshape = pim_gemm_desc->out_bshape_r;

    pim_gemm_desc->in_bshape.w = aligned_in_w;
    pim_gemm_desc->wei_bshape.h = aligned_in_w;
    pim_gemm_desc->wei_bshape.w = aligned_out_w;
    pim_gemm_desc->bias_bshape.w = aligned_out_w;
    pim_gemm_desc->out_bshape.w = aligned_out_w;
}

void align_shape(PimDesc* pim_desc, PimOpType op_type)
{
    PimBShape bs = pim_desc->bshape_r;

    if (op_type == OP_GEMV) {
        bs.h = PIM_GEMV_IN_ALIGN * ceil((float)bs.h / PIM_GEMV_IN_ALIGN);
        bs.w = PIM_GEMV_OUT_ALIGN * ceil((float)bs.w / PIM_GEMV_OUT_ALIGN);
    } else {
        bs.w = PIM_ELTWISE_ALIGN * ceil((float)bs.w / PIM_ELTWISE_ALIGN);
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

bool check_chwise_gemm_bo(PimBo* bo)
{
    bool ret = true;

    if (bo->bshape.w >= PIM_GEMV_OUT_ALIGN) return false;
    if ((bo->bshape.n * bo->bshape.c * bo->bshape.w) % PIM_GEMV_OUT_ALIGN != 0) return false;

    return ret;
}

bool is_pim_available(PimBo* out, PimBo* op0, PimBo* op1, PimOpType op_type)
{
    bool ret = false;

    switch (op_type) {
        case OP_GEMV:
            ret = is_pim_gemv_available(op1);
            break;
        default:
            ret = false;
    }

    return ret;
}

bool is_pim_gemv_available(PimBo* bo)
{
    /* TODO: find optimal shape to execute PIM ops */
    if (bo->bshape_r.n * bo->bshape_r.c * bo->bshape_r.w >= DIM_OUT_PIM)
        return true;
    else
        return false;
}

bool is_pim_gemv_list_available(PimBo* output, PimBo* vector, PimBo* matrix)
{
    int out_dim = output->size;

    if ((out_dim % PIM_GEMV_OUT_ALIGN) != 0) return false;
    if (output->bshape.c == 1) return false;
    if (output->bshape.c != vector->bshape.c || output->bshape.c != matrix->bshape.c) return false;

    return true;
}

size_t PrecisionSize(const PimBo* bo)
{
    size_t ret = 0;
    assert(bo != nullptr && "Invalid buffer");
    switch (bo->precision) {
        case PIM_FP16:
            ret = sizeof(half_float::half);
        case PIM_INT8:
        default:
            ret = 1ul;
    }
    return ret;
}
