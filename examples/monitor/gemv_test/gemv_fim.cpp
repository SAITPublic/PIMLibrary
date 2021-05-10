/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <stdio.h>
#include <iostream>
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

#define PARK_IN 1
#define CHANGE_SB_HAB 1
#define PROGRAM_CRF 1
#define COMPUTE_GEMV 1
#define CHANGE_HAB_SB 1
#define PARK_OUT 1
#define INTEGRAL_SUM 1

#define RADEON7 0

extern "C" uint64_t fmm_map_pim(uint32_t, uint32_t, uint64_t);

#define CHECK(cmd)                                                                                              \
    {                                                                                                           \
        hipError_t error = cmd;                                                                                 \
        if (error != hipSuccess) {                                                                              \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    }

__device__ inline void R_CMD(volatile uint8_t* src)
{
    asm volatile("global_load_dwordx4 v[24:27], %0, off, glc, slc" ::"v"(src) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD(volatile uint8_t* dst)
{
    asm volatile("global_store_dwordx4 %0, v[24:27], off, glc, slc" ::"v"(dst) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD_R(volatile uint8_t* dst, volatile uint8_t* src) { ((int4*)dst)[0] = ((int4*)src)[0]; }

__device__ inline void B_CMD(int type)
{
    if (type == 0) {
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    } else {
        __threadfence();
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    }
}

__device__ inline unsigned int mask_by_bit(unsigned int value, int start, int end)
{
    int length = start - end + 1;
    value = value >> end;
    return value & ((1 << length) - 1);
}

__device__ uint64_t addr_gen(unsigned int ch, unsigned int rank, unsigned int bg, unsigned int ba, unsigned int row,
                             unsigned int col)
{
    int num_row_bit_ = 14;
    int num_col_high_bit_ = 3;
    int num_bank_high_bit_ = 1;
    int num_bankgroup_bit_ = 2;
    int num_bank_low_bit_ = 1;
    int num_chan_bit_ = 6;
    int num_col_low_bit_ = 1;
    int num_offset_bit_ = 5;

    uint64_t addr = rank;

    addr <<= num_row_bit_;
    addr |= row;

    addr <<= num_col_high_bit_;
    addr |= mask_by_bit(col, 4, 2);  // HARDCODED. FIXME

    addr <<= num_bank_high_bit_;
    addr |= mask_by_bit(ba, 1, 1);

    addr <<= num_bankgroup_bit_;
    addr |= bg;

    addr <<= num_bank_low_bit_;
    addr |= mask_by_bit(ba, 0, 0);

    addr <<= num_chan_bit_ - 1;
    addr |= mask_by_bit(ch, num_chan_bit_ - 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(col, 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(ch, 0, 0);

    addr <<= 1;
    addr |= mask_by_bit(col, 0, 0);

    addr <<= num_offset_bit_;

#if RADEON7
    uint64_t mask = 0x1FFFFFFFF;
    addr &= mask;
#endif
    return addr;
}

__global__ void gemv_pim_64cu_64th_fp16(volatile uint8_t* pim_ctr, volatile uint8_t* pim_weight,
                                        volatile uint8_t* pim_gemv_tmp_buffer, volatile uint8_t* pim_input,
                                        volatile uint8_t* output, int batch_dim, int n_in_tile, int n_out_tile,
                                        int output_dim, volatile uint8_t* gemv_crf, volatile uint8_t* hab_to_pim,
                                        volatile uint8_t* pim_to_hab)
{
    int num_col = 32;
    int num_bg = 4;
    int num_ba = 4;
    int num_grf = 8;
    int trans_size = 32;
    int loc, row, col;

    int num_in_tile = n_in_tile;
    int num_out_tile = n_out_tile;

    int gidx = hipThreadIdx_x >> 1;
    uint64_t offset = (hipThreadIdx_x % 2) << 4;
    uint64_t addr;

#if PARK_IN
    /* num_bg * num_ba * 2 */
    if (hipThreadIdx_x < 32) {
        addr = addr_gen(hipBlockIdx_x, 0, gidx >> 2, gidx % num_ba, 0, 0);
        R_CMD(&pim_ctr[addr + offset]);
    }
    B_CMD(0);
#endif

#if CHANGE_SB_HAB
    if (hipThreadIdx_x < 2) {
        addr = addr_gen(hipBlockIdx_x, 0, 2, gidx, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen(hipBlockIdx_x, 0, 2, gidx + 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen(hipBlockIdx_x, 0, 0, gidx, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen(hipBlockIdx_x, 0, 0, gidx + 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
    }
    B_CMD(0);
#endif

#if PROGRAM_CRF
    if (hipThreadIdx_x < 2 /* sizeof(gemv_crf) >> 4 */) {
        addr = addr_gen(hipBlockIdx_x, 0, 0, 1, 0x3fff, 0x4 + gidx);
        W_CMD_R(&pim_ctr[addr + offset], gemv_crf + (hipThreadIdx_x << 4));
    }
    B_CMD(0);
#endif

#if COMPUTE_GEMV
    if (hipThreadIdx_x < 16 /* 2 * num_grf */) {
        for (int b_idx = 0; b_idx < batch_dim; b_idx++) {
            for (int o_idx = 0; o_idx < num_out_tile; o_idx++) {
                addr = addr_gen(hipBlockIdx_x, 0, 0, 0, 0x3fff, 0x0);
                W_CMD_R(&pim_ctr[addr + offset], hab_to_pim + ((hipThreadIdx_x % 2) << 4));
                B_CMD(1);

                uint64_t i_offset = (b_idx * num_in_tile << 3) + gidx;
                int r_offset = (o_idx * num_in_tile) >> 1;

                for (int i_idx = 0; i_idx < num_in_tile; i_idx += 2) {
                    uint64_t i_addr = (i_offset + (i_idx << 3)) << 5;
                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, 0x3fff, 0x8 + gidx);
                    W_CMD_R(&pim_ctr[addr + offset], &pim_input[i_addr + offset]);
                    B_CMD(1);

                    row = ((i_idx >> 1) + r_offset) << 1;
                    col = gidx;

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col + 8);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col + 16);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col + 24);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row + 1, col);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row + 1, col + 8);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row + 1, col + 16);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row + 1, col + 24);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);
                }

                for (int i_idx = 1; i_idx < num_in_tile; i_idx += 2) {
                    uint64_t i_addr = (i_offset + (i_idx << 3)) << 5;
                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, 0x3fff, 0x8 + gidx);
                    W_CMD_R(&pim_ctr[addr + offset], &pim_input[i_addr + offset]);
                    B_CMD(1);

                    row = ((i_idx >> 1) + r_offset) << 1;
                    col = gidx;

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col + 8);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col + 16);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col + 24);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row + 1, col);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row + 1, col + 8);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row + 1, col + 16);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row + 1, col + 24);
                    R_CMD(&pim_weight[addr + offset]);
                    B_CMD(1);
                }
                loc = b_idx * num_out_tile * num_grf + o_idx * num_grf + gidx;
                row = loc >> 5;  // loc / num_col
                col = loc % num_col;

                addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col);
                W_CMD(&pim_gemv_tmp_buffer[addr + offset]);
                B_CMD(1);

                addr = addr_gen(hipBlockIdx_x, 0, 0, 0, 0x3fff, 0x0);
                W_CMD_R(&pim_ctr[addr + offset], pim_to_hab + ((hipThreadIdx_x % 2) << 4));
                B_CMD(1);
            }
        }
    }
    B_CMD(0);
#endif

#if CHANGE_HAB_SB
    if (hipThreadIdx_x < 4) {
        addr = addr_gen(hipBlockIdx_x, 0, 0, gidx, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
    }
    B_CMD(0);
#endif

#if PARK_OUT
    if (hipThreadIdx_x < 32) {
        addr = addr_gen(hipBlockIdx_x, 0, gidx / num_ba, gidx % num_ba, 0, 0);
        R_CMD(&pim_weight[addr + offset]);
    }
    B_CMD(0);
#endif

#if INTEGRAL_SUM
    int bg = hipThreadIdx_x >> 4;
    int ba = (((hipThreadIdx_x >> 3) % 2) << 1) + 1;
    int out_idx = (hipBlockIdx_x << 6) + hipThreadIdx_x;
    uint16_t t_output;
    for (int i = 0; i < batch_dim * num_out_tile; i++) {
        if (out_idx < output_dim) {
            t_output = 0;
            row = i >> 2;
            col = hipThreadIdx_x % 8 + ((i % 4) << 3);
            addr = addr_gen(hipBlockIdx_x, 0, bg, ba, row, col);
            for (int j = 0; j < 16; j++) {
                t_output += pim_gemv_tmp_buffer[addr + j];
            }
            ((uint16_t*)output)[out_idx] = ((uint16_t*)output)[out_idx] + t_output;
        }
    }
#endif
}

int main(int argc, char* argv[])
{
    uint64_t pim_base;
    uint16_t* input;
    uint16_t* output;
    uint64_t *mode1_d, *mode2_d, *crf_bin_d;
    uint64_t *mode1_h, *mode2_h, *crf_bin_h;
    size_t N = 4;
    size_t Nbytes = N * sizeof(uint64_t);
    static int device = 0;

    int num_pim_blocks = 8;
    int num_pim_chan = 64;
    int num_grf = 8;
    int trans_size = 32;

    int in_size = 512;
    int out_size = 4096;

    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));

    // Get GPU ID
    FILE* fd;
    char path[256];
    uint32_t gpu_id;

    snprintf(path, 256, "/sys/devices/virtual/kfd/kfd/topology/nodes/1/gpu_id");
    fd = fopen(path, "r");
    if (!fd) return -1;
    if (fscanf(fd, "%ul", &gpu_id) != 1) return -1;
    fclose(fd);

    uint64_t ret = 0;
    /********************************************
      ARG1 : node-id
      ARG2 : gpu-id
      ARG3 : block size
    ********************************************/
#if RADEON7
    uint64_t bsize = 8589934592;  // 8 * 1024 * 1024 * 1024;
#else
    uint64_t bsize = 17179869184;  // 16 * 1024 * 1024 * 1024;
#endif
    pim_base = fmm_map_pim(1, gpu_id, bsize);
    std::cout << std::hex << "pimBaseAddr = " << pim_base << std::endl;

    CHECK(hipMalloc(&input, in_size * sizeof(uint16_t)));
    CHECK(hipMalloc(&output, out_size * sizeof(uint16_t)));

    crf_bin_h = (uint64_t*)malloc(Nbytes);
    CHECK(crf_bin_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    mode1_h = (uint64_t*)malloc(Nbytes);
    CHECK(mode1_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    mode2_h = (uint64_t*)malloc(Nbytes);
    CHECK(mode2_h == 0 ? hipErrorOutOfMemory : hipSuccess);

    crf_bin_h[0] = 0xe00078023b108000;
    crf_bin_h[1] = 0xe00078023b188000;
    crf_bin_h[2] = 0xf000000000000007;
    crf_bin_h[3] = 0x0000000000000000;
    mode1_h[0] = 0x0000000000000001;
    mode1_h[1] = 0x0000000000000000;
    mode1_h[2] = 0x0000010000000000;
    mode1_h[3] = 0x0000000000000000;
    mode2_h[0] = 0x0000000000000000;
    mode2_h[1] = 0x0000000000000000;
    mode2_h[2] = 0x0000000000000000;
    mode2_h[3] = 0x0000000000000000;

    CHECK(hipMalloc(&crf_bin_d, Nbytes));
    CHECK(hipMalloc(&mode1_d, Nbytes));
    CHECK(hipMalloc(&mode2_d, Nbytes));

    CHECK(hipMemcpy(crf_bin_d, crf_bin_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode1_d, mode1_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode2_d, mode2_h, Nbytes, hipMemcpyHostToDevice));

    int n_in_tile = in_size * sizeof(uint16_t) / trans_size / num_grf;
    int n_out_tile = out_size / (num_pim_chan * num_pim_blocks * num_grf);

    const unsigned blocks = 64;
    const unsigned threadsPerBlock = 64;

    hipLaunchKernelGGL(gemv_pim_64cu_64th_fp16, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       (uint8_t*)pim_base /* ctr base */, (uint8_t*)pim_base /* weight */,
                       (uint8_t*)pim_base + 0x200000, /* pim hw output */
                       (uint8_t*)input, (uint8_t*)output, 1, n_in_tile, n_out_tile, out_size, (uint8_t*)crf_bin_d,
                       (uint8_t*)mode1_d, (uint8_t*)mode2_d);
    hipStreamSynchronize(NULL);

    free(mode1_h);
    free(mode2_h);
    free(crf_bin_h);

    hipFree(mode1_d);
    hipFree(mode2_d);
    hipFree(crf_bin_d);
    hipFree(input);
    hipFree(output);
}
