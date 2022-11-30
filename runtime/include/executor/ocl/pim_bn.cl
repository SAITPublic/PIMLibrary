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

#ifndef _PIM_BN_KERNELS_PIMK_
#define _PIM_BN_KERNELS_PIMK_

#define PARK_IN 1
#define PROGRAM_SRF 1
#define CHANGE_SB_HAB 1
#define PROGRAM_CRF 1
#define CHANGE_HAB_HABPIM 1
#define COMPUTE_BN 1
#define CHANGE_HABPIM_HAB 1
#define CHANGE_HAB_SB 1
#define PARK_OUT 1

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

#if PARK_IN
    park_in(pim_ctr, gidx, num_bank, offset);
#endif

    if (get_local_id(0) < 2) {
#if CHANGE_SB_HAB
        change_sb_hab(pim_ctr, offset);
#endif

#if PROGRAM_SRF
        program_srf(pim_ctr, srf_binary, offset);
#endif
#if CHANGE_HAB_HABPIM
        change_hab_habpim(pim_ctr, offset);
        B_CMD(1);
#endif
    }

#if PROGRAM_CRF
    if (get_local_id(0) < (crf_size >> 4)) {
        program_crf_mod(pim_ctr, gidx, crf_binary, offset);
    }
#endif

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
#if CHANGE_HABPIM_HAB
        change_habpim_hab(pim_ctr, offset);
#endif

#if CHANGE_HAB_SB
        change_hab_sb(pim_ctr, gidx, offset);
#endif
    }

#if PARK_OUT
    park_out(pim_ctr, gidx, num_bank, offset);
#endif

#ifdef EMULATOR
    if (get_group_id(0) == 0 && get_local_id(0) == 0) {
        frd_size[0] = emulator_trace->g_ridx[0];
    }
#endif
}

#endif /* _PIM_BN_KERNELS_PIMK_ */
