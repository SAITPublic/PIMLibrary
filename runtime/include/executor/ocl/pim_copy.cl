/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_COPY_KERNELS_PIMK_
#define _PIM_COPY_KERNELS_PIMK_

#define COMPUTE_COPY 1

__kernel void copy_pim(__global uint8_t* __restrict__ pim_data, __global uint8_t* __restrict__ output,
                       __global uint8_t* __restrict__ pim_ctr, int size, __global uint8_t* crf_binary, int crf_size
#ifdef EMULATOR
                       ,
                       __global PimMemTraceData* fmtd16, __global size_t* frd_size, int mt_width,
                       __global PimMemTracer* emulator_trace
#endif
                       )
{
#ifdef EMULATOR
    emulator_trace->g_fba = (uint64_t)pim_ctr;
    emulator_trace->g_fmtd16 = fmtd16;
    emulator_trace->g_ridx[get_group_id(0)] = 0;
    emulator_trace->m_width = mt_width;
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    int trans_size = 32;
    int num_col = 32;
    int num_pim_blocks = 8;
    int num_pim_chan = 64;
    int num_grf = 8;
    int num_ba = 4;
    int out_dim = size / trans_size;
    int num_tile = out_dim / (num_pim_blocks * num_pim_chan * num_grf) / 2;

    int gidx = get_local_id(0) / 2;
    uint64_t offset = (get_local_id(0) % 2) * 0x10;
    uint64_t addr, addr_even, addr_odd;

/* Radeon7(VEGA20) memory is 16GB but our target is 32GB system */
/* so program_crf and chagne_pim_mode functions can not access to over 8GB in our system */

#if PARK_IN
    addr = addr_gen_(get_group_id(0), 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
#endif

    if (get_local_id(0) < 2) {
#if CHANGE_SB_HAB
        addr = addr_gen_(get_group_id(0), 0, 2, 0, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 2, 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
#if PROGRAM_CRF
        addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x3fff, 0x4 + gidx);
        W_CMD_R(&pim_ctr[addr + offset], crf_binary + (get_local_id(0) << 4));
#endif
#if CHANGE_HAB_HABPIM
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R_C(&pim_ctr[addr + offset], elt_add_hab_to_hab_pim + offset);
        R_CMD(&pim_ctr[addr + offset]);
#endif
        B_CMD(1);
    }

#if COMPUTE_COPY
    if (get_local_id(0) < 16) {
        for (int tile_idx = 0; tile_idx < num_tile; tile_idx++) {
            unsigned int loc = tile_idx * num_grf + gidx;
            unsigned int row = loc / num_col;
            unsigned int col = loc % num_col;

            addr = addr_gen_(get_group_id(0), 0, 0, 0, row, col);
            addr_even = addr + offset;
            addr_odd = addr_even + 0x2000;

            R_CMD(&pim_data[addr_even]);
            B_CMD(1);

            W_CMD(&output[addr_even]);
            R_CMD(&output[addr_even]);
            B_CMD(1);

            R_CMD(&pim_data[addr_odd]);
            B_CMD(1);

            W_CMD(&output[addr_odd]);
            R_CMD(&output[addr_odd]);
            B_CMD(1);
        }
    }
#endif

    if (get_local_id(0) < 4) {
#if CHANGE_HABPIM_HAB
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R_C(&pim_ctr[addr + offset], elt_add_hab_pim_to_hab + offset);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
#if CHANGE_HAB_SB
        addr = addr_gen_(get_group_id(0), 0, 0, gidx, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
    }

#if PARK_OUT
    addr = addr_gen_(get_group_id(0), 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
    W_CMD(&pim_ctr[addr + offset]);
#endif

#ifdef EMULATOR
    if (get_group_id(0) == 0 && get_local_id(0) == 0) {
        frd_size[0] = emulator_trace->g_ridx[0];
    }
#endif
}

#endif /* _PIM_COPY_KERNELS_PIMK_ */
