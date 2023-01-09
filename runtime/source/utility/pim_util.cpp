/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "utility/pim_util.h"
#include <cassert>
#include "half.hpp"
// using namespace half_float::half;

#if AMD
__host__ void get_pim_block_info(PimBlockInfo* pbi) { memcpy(pbi, &vega20_pbi, sizeof(PimBlockInfo)); }
#else
void get_pim_block_info(PimBlockInfo* pbi) { memcpy(pbi, &vega20_pbi, sizeof(PimBlockInfo)); }
#endif

void set_pimbo_t(PimBo* bo0, PimBo* bo1, PimBo* bo2, PimBo* bo3)
{
    set_pimbo_t(bo0);
    set_pimbo_t(bo1);
    set_pimbo_t(bo2);
    set_pimbo_t(bo3);
}

void set_pimbo_t(PimBo* inout)
{
    if (inout == nullptr) return;
    uint32_t h = inout->bshape.h;
    uint32_t w = inout->bshape.w;
    uint32_t h_r = inout->bshape_r.h;
    uint32_t w_r = inout->bshape_r.w;

    inout->bshape.h = w;
    inout->bshape.w = h;
    inout->bshape_r.h = w_r;
    inout->bshape_r.w = h_r;
}

void set_pimbo_t(PimBo* dst, PimBo* src)
{
    if (src == nullptr) return;
    if (dst == nullptr) return;
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

void set_pimbo(PimGemmDesc* pim_gemm_desc, PimMemType mem_type, PimMemFlag mem_flag, bool transposed, PimBo* pim_bo)
{
    size_t size = 0;
    size_t size_r = 0;

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
    size_r = bshape_r->n * bshape_r->c * bshape_r->h * bshape_r->w;
    size = bshape->n * bshape->c * bshape->h * bshape->w;
    if (pim_gemm_desc->precision == PIM_FP16) {
        size *= 2;
        size_r *= 2;
    }

    pim_bo->size = size;
    pim_bo->size_r = size_r;
    pim_bo->mem_type = mem_type;
    pim_bo->precision = pim_gemm_desc->precision;
    pim_bo->data_layout_type = PimDataLayoutType::RAW;
    pim_bo->bshape = *bshape;
    pim_bo->bshape_r = *bshape_r;
    pim_bo->transposed = transposed;
}

void align_gemm_shape(PimGemmDesc* pim_gemm_desc)
{
    int n = pim_gemm_desc->in_bshape_r.n;
    int c = pim_gemm_desc->in_bshape_r.c;
    int in_h = pim_gemm_desc->in_bshape_r.h;
    int in_w = pim_gemm_desc->in_bshape_r.w;
    int out_h = pim_gemm_desc->out_bshape_r.h;
    int out_w = pim_gemm_desc->out_bshape_r.w;
    int aligned_in_size = 0;
    int aligned_out_size = 0;
    int out_align = 0;

    pim_gemm_desc->in_bshape = pim_gemm_desc->in_bshape_r;
    pim_gemm_desc->wei_bshape = pim_gemm_desc->wei_bshape_r;
    pim_gemm_desc->bias_bshape = pim_gemm_desc->bias_bshape_r;
    pim_gemm_desc->out_bshape = pim_gemm_desc->out_bshape_r;

    if (pim_gemm_desc->gemm_order == W_X_I) {
        aligned_in_size = PIM_GEMV_IN_ALIGN * ceil((float)in_h / PIM_GEMV_IN_ALIGN);
        out_align = n * c * out_h;
        if (out_align % PIM_GEMV_OUT_ALIGN == 0)
            aligned_out_size = out_h;
        else
            aligned_out_size = PIM_GEMV_OUT_ALIGN * ceil((float)out_h / PIM_GEMV_OUT_ALIGN);

        pim_gemm_desc->in_bshape.h = aligned_in_size;
        pim_gemm_desc->wei_bshape.w = aligned_in_size;
        pim_gemm_desc->wei_bshape.h = aligned_out_size;
        pim_gemm_desc->bias_bshape.h = aligned_out_size;
        pim_gemm_desc->out_bshape.h = aligned_out_size;
    } else {
        aligned_in_size = PIM_GEMV_IN_ALIGN * ceil((float)in_w / PIM_GEMV_IN_ALIGN);
        out_align = n * c * out_w;
        if (out_align % PIM_GEMV_OUT_ALIGN == 0)
            aligned_out_size = out_w;
        else
            aligned_out_size = PIM_GEMV_OUT_ALIGN * ceil((float)out_w / PIM_GEMV_OUT_ALIGN);

        pim_gemm_desc->in_bshape.w = aligned_in_size;
        pim_gemm_desc->wei_bshape.h = aligned_in_size;
        pim_gemm_desc->wei_bshape.w = aligned_out_size;
        pim_gemm_desc->bias_bshape.w = aligned_out_size;
        pim_gemm_desc->out_bshape.w = aligned_out_size;
    }
}

void align_shape(PimDesc* pim_desc, PimOpType op_type)
{
    PimBShape bs = pim_desc->bshape_r;

    if (op_type == OP_GEMV) {
        bs.h = PIM_GEMV_IN_ALIGN * ceil((float)bs.h / PIM_GEMV_IN_ALIGN);
        bs.w = PIM_GEMV_OUT_ALIGN * ceil((float)bs.w / PIM_GEMV_OUT_ALIGN);
    } else {
        const auto buffer_length = bs.n * bs.c * bs.h * bs.w;
        const auto aligned_length = PIM_ELTWISE_ALIGN * ceil((float)buffer_length / PIM_ELTWISE_ALIGN);
        // Align width to cover aligned length
        bs.w = ceil((float)aligned_length / (bs.n * bs.h * bs.c));
    }
    pim_desc->bshape = bs;
}

// TODO gneralise for hip and ocl.
void pad_data(void* input, PimDesc* pim_desc, PimMemType mem_type, PimMemFlag mem_flag)
{
    if (mem_flag == GEMV_INPUT && mem_type == MEM_TYPE_HOST) {
        if (mem_type == MEM_TYPE_HOST) {
            for (int i = 0; i < pim_desc->bshape.n; i++) {
                for (int j = pim_desc->bshape_r.w; j < pim_desc->bshape.w; j++) {
                    ((half_float::half*)input)[i * pim_desc->bshape.w + j] = half_float::half(0);
                }
            }
        } else {
            if (pim_desc->bshape_r.w != pim_desc->bshape.w) {
#if AMD
                hipMemset(input, 0, pim_desc->bshape.n * pim_desc->bshape.w * sizeof(half));
#endif
            }
        }
    }

    if (mem_flag == ELT_OP) {
        int padded_size = pim_desc->bshape.n * pim_desc->bshape.c * pim_desc->bshape.h * pim_desc->bshape.w;
        int real_size = pim_desc->bshape_r.n * pim_desc->bshape_r.c * pim_desc->bshape_r.h * pim_desc->bshape_r.w;
        for (int i = real_size; i < padded_size; i++) {
            ((half_float::half*)input)[i] = half_float::half(0);
        }
    }
}

void pad_data(void* input, int in_size, int in_nsize, int batch_size, PimMemFlag mem_flag)
{
    if (mem_flag == GEMV_INPUT) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = in_size; j < in_nsize; j++) {
                ((half_float::half*)input)[in_nsize * i + j] = half_float::half(0);
            }
        }
    } else {
        for (int i = in_size; i < in_nsize; i++) {
            ((half_float::half*)input)[i] = half_float::half(0);
        }
    }
}

bool check_chwise_gemm_bo(PimBo* bo, PimGemmOrder gemm_order)
{
    bool ret = true;

    if (gemm_order == I_X_W) {
        if (bo->bshape.w >= PIM_GEMV_OUT_ALIGN) return false;
        if ((bo->bshape.n * bo->bshape.c * bo->bshape.w) % PIM_GEMV_OUT_ALIGN != 0) return false;
    } else {
        if (bo->bshape.h >= PIM_GEMV_OUT_ALIGN) return false;
        if ((bo->bshape.n * bo->bshape.c * bo->bshape.h) % PIM_GEMV_OUT_ALIGN != 0) return false;
    }

    return ret;
}

bool is_pim_applicable(PimBo* wei, PimGemmOrder gemm_order)
{
    if (wei->data_layout_type != PimDataLayoutType::RAW) {
        // PIM-specific layout has been already created
        return true;
    }
    /* TODO: find optimal shape to execute PIM ops */
    uint32_t wei_dim = (gemm_order == PimGemmOrder::W_X_I) ? wei->bshape_r.h : wei->bshape_r.w;

    /* TODO: This weight dim is only support in CGEMV */
    if (wei_dim == 32317) return false;

    if (wei->bshape_r.n * wei->bshape_r.c * wei_dim >= DIM_OUT_PIM)
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
            break;
        case PIM_INT8:
        default:
            ret = 1ul;
    }
    return ret;
}
