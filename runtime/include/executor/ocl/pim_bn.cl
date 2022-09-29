/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_BN_KERNELS_PIMK_
#define _PIM_BN_KERNELS_PIMK_
#define COMPUTE_BN 1

static __constant uint8_t bn_hab_pim_to_hab[32] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

static __constant uint8_t bn_hab_to_hab_pim[32] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

__kernel void bn_pim_nr_sip(__global uint8_t* __restrict__ pim_data, __global uint8_t* __restrict__ output,
                            __global uint8_t* __restrict__ pim_ctr, int num_tile, __global uint8_t* crf_binary,
                            int crf_size, __global uint8_t* srf_binary, int srf_size
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
    int num_col = 32;
    int num_grf = 8;
    int num_bank = 4;

    int gidx = get_local_id(0) / 2;
    uint64_t offset = (get_local_id(0) % 2) * 0x10;
    uint64_t addr, addr_even, addr_odd;

    /* Radeon7(VEGA20) memory is 16GB but our target is 32GB system */
    /* so program_crf and chagne_pim_mode functions can not access to over 8GB in our system */

    addr = addr_gen_(get_group_id(0), 0, gidx / num_bank, gidx % num_bank, 1 << 13, 0);
    W_CMD(&pim_data[addr + offset]);
    B_CMD(1);

    if (get_local_id(0) < 2) {
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

        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R(&pim_ctr[addr + 32 + offset], srf_binary + offset);
        W_CMD_R_C(&pim_ctr[addr + offset], bn_hab_to_hab_pim + offset);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
    }

    if (get_local_id(0) < (crf_size >> 4)) {
        addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x3fff, 0x4 + gidx);
        W_CMD_R(&pim_ctr[addr + offset], crf_binary + (get_local_id(0) << 4));
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
    }

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

            R_CMD(&pim_data[addr_even]);
            B_CMD(1);

            R_CMD(&pim_data[addr_even]);
            B_CMD(1);

            R_CMD(&pim_data[addr_even]);
            B_CMD(1);

            R_CMD(&pim_data[addr_odd]);
            B_CMD(1);

            R_CMD(&pim_data[addr_odd]);
            B_CMD(1);

            R_CMD(&pim_data[addr_odd]);
            B_CMD(1);

            W_CMD(&output[addr_even]);
            B_CMD(1);

            W_CMD(&output[addr_odd]);
            R_CMD(&output[addr_odd]);
            B_CMD(1);
        }
    }

    if (get_local_id(0) < 4) {
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R_C(&pim_ctr[addr + offset], bn_hab_pim_to_hab + offset);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);

        addr = addr_gen_(get_group_id(0), 0, 0, gidx, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
    }

    addr = addr_gen_(get_group_id(0), 0, gidx / num_bank, gidx % num_bank, 1 << 13, 0);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);

#ifdef EMULATOR
    if (get_group_id(0) == 0 && get_local_id(0) == 0) {
        frd_size[0] = emulator_trace->g_ridx[0];
    }
#endif
}

#endif /* _PIM_BN_KERNELS_PIMK_ */
