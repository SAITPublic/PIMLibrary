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

extern "C" uint64_t fmm_map_fim(uint32_t, uint32_t, uint64_t);

#define CHECK(cmd)                                                                                              \
    {                                                                                                           \
        hipError_t error = cmd;                                                                                 \
        if (error != hipSuccess) {                                                                              \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    }

__device__ inline void R_CMD(uint8_t* src)
{
    asm volatile("global_load_dwordx4 v[24:27], %0, off, glc, slc" ::"v"(src) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD(uint8_t* dst)
{
    asm volatile("global_store_dwordx4 %0, v[24:27], off, glc, slc" ::"v"(dst) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD_R(uint8_t* dst, uint8_t* src) { ((int4*)dst)[0] = ((int4*)src)[0]; }

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

    return addr;
}

uint8_t gemv_crf[32] = {
    0x00, 0x80, 0x10, 0x3b, 0x02, 0x38, 0x00, 0xe0, 0x00, 0x80, 0x18, 0x3b, 0x02, 0x38, 0x00, 0xe0,
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

uint8_t gemv_hab_to_hab_fim[32] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

uint8_t gemv_hab_fim_to_hab[32] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

__global__ void gemv_fim_64cu_64th_fp16(uint8_t* fim_ctr, uint8_t* fim_weight, uint8_t* fim_gemv_tmp_buffer,
                                        uint8_t* fim_input, uint8_t* output, int batch_dim, int n_in_tile,
                                        int n_out_tile, int output_dim)
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
        R_CMD(&fim_ctr[addr + offset]);
    }
    B_CMD(0);
#endif

#if CHANGE_SB_HAB
    if (hipThreadIdx_x < 2) {
        addr = addr_gen(hipBlockIdx_x, 0, 2, gidx, 0x27ff, 0x1f);
        W_CMD(&fim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen(hipBlockIdx_x, 0, 2, gidx + 1, 0x27ff, 0x1f);
        W_CMD(&fim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen(hipBlockIdx_x, 0, 0, gidx, 0x27ff, 0x1f);
        W_CMD(&fim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen(hipBlockIdx_x, 0, 0, gidx + 1, 0x27ff, 0x1f);
        W_CMD(&fim_ctr[addr + offset]);
        B_CMD(1);
    }
    B_CMD(0);
#endif

#if PROGRAM_CRF
    if (hipThreadIdx_x < 2 /* sizeof(gemv_crf) >> 4 */) {
        addr = addr_gen(hipBlockIdx_x, 0, 0, 1, 0x3fff, 0x4 + gidx);
        W_CMD_R(&fim_ctr[addr + offset], gemv_crf + (hipThreadIdx_x << 4));
    }
    B_CMD(0);
#endif

#if COMPUTE_GEMV
    if (hipThreadIdx_x < 16 /* 2 * num_grf */) {
        for (int b_idx = 0; b_idx < batch_dim; b_idx++) {
            for (int o_idx = 0; o_idx < num_out_tile; o_idx++) {
                addr = addr_gen(hipBlockIdx_x, 0, 0, 0, 0x3fff, 0x0);
                W_CMD_R(&fim_ctr[addr + offset], gemv_hab_to_hab_fim + ((hipThreadIdx_x % 2) << 4));
                B_CMD(1);

                uint64_t i_offset = (b_idx * num_in_tile << 3) + gidx;
                int r_offset = (o_idx * num_in_tile) >> 1;

                for (int i_idx = 0; i_idx < num_in_tile; i_idx += 2) {
                    uint64_t i_addr = (i_offset + (i_idx << 3)) << 5;
                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, 0x3fff, 0x8 + gidx);
                    W_CMD_R(&fim_ctr[addr + offset], &fim_input[i_addr + offset]);
                    B_CMD(1);

                    row = ((i_idx >> 1) + r_offset) << 1;
                    col = gidx;

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col + 8);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col + 16);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col + 24);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row + 1, col);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row + 1, col + 8);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row + 1, col + 16);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row + 1, col + 24);
                    R_CMD(&fim_weight[addr + offset]);
                    B_CMD(1);
                }

                for (int i_idx = 1; i_idx < num_in_tile; i_idx += 2) {
                    uint64_t i_addr = (i_offset + (i_idx << 3)) << 5;
                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, 0x3fff, 0x8 + gidx);
                    W_CMD_R(&fim_ctr[addr + offset], &fim_input[i_addr + offset]);
                    B_CMD(1);

                    row = ((i_idx >> 1) + r_offset) << 1;
                    col = gidx;

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col + 8);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col + 16);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col + 24);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row + 1, col);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row + 1, col + 8);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row + 1, col + 16);
                    R_CMD(&fim_weight[addr + offset]);

                    addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row + 1, col + 24);
                    R_CMD(&fim_weight[addr + offset]);
                    B_CMD(1);
                }
                loc = b_idx * num_out_tile * num_grf + o_idx * num_grf + gidx;
                row = loc >> 5;  // loc / num_col
                col = loc % num_col;

                addr = addr_gen(hipBlockIdx_x, 0, 0, 1, row, col);
                W_CMD(&fim_gemv_tmp_buffer[addr + offset]);
                B_CMD(1);

                addr = addr_gen(hipBlockIdx_x, 0, 0, 0, 0x3fff, 0x0);
                W_CMD_R(&fim_ctr[addr + offset], gemv_hab_fim_to_hab + ((hipThreadIdx_x % 2) << 4));
                B_CMD(1);
            }
        }
    }
    B_CMD(0);
#endif

#if CHANGE_HAB_SB
    if (hipThreadIdx_x < 4) {
        addr = addr_gen(hipBlockIdx_x, 0, 0, gidx, 0x2fff, 0x1f);
        W_CMD(&fim_ctr[addr + offset]);
    }
    B_CMD(0);
#endif

#if PARK_OUT
    if (hipThreadIdx_x < 32) {
        addr = addr_gen(hipBlockIdx_x, 0, gidx / num_ba, gidx % num_ba, 0, 0);
        R_CMD(&fim_weight[addr + offset]);
    }
    B_CMD(0);
#endif

#if INTEGRAL_SUM
    int bg = hipThreadIdx_x >> 4;
    int ba = (((hipThreadIdx_x >> 3) % 2) << 1) + 1;
    int out_idx = (hipBlockIdx_x << 6) + hipThreadIdx_x;
    half t_output;
    for (int i = 0; i < batch_dim * num_out_tile; i++) {
        if (out_idx < output_dim) {
            t_output = 0;
            row = i >> 2;
            col = hipThreadIdx_x % 8 + ((i % 4) << 3);
            addr = addr_gen(hipBlockIdx_x, 0, bg, ba, row, col);
            for (int j = 0; j < 16; j++) {
                t_output += fim_gemv_tmp_buffer[addr + j];
            }
            ((half*)output)[out_idx] = ((half*)output)[out_idx] + t_output;
        }
    }
#endif
}

int main(int argc, char* argv[])
{
    uint64_t fim_base;
    uint16_t* input;
    uint16_t* output;
    size_t N = 4;
    size_t Nbytes = N * sizeof(uint64_t);
    static int device = 0;

    int num_fim_blocks = 8;
    int num_fim_chan = 64;
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
    //uint64_t bsize = 8589934592;  // 8 * 1024 * 1024 * 1024;
    uint64_t bsize = 17179869184;  // 16 * 1024 * 1024 * 1024;
    fim_base = fmm_map_fim(1, gpu_id, bsize);
    std::cout << std::hex << "fimBaseAddr = " << fim_base << std::endl;

    CHECK(hipMalloc(&input, in_size * sizeof(uint16_t)));
    CHECK(hipMalloc(&output, out_size * sizeof(uint16_t)));

    int n_in_tile = in_size * sizeof(uint16_t) / trans_size / num_grf;
    int n_out_tile = out_size / (num_fim_chan * num_fim_blocks * num_grf);

    const unsigned blocks = 64;
    const unsigned threadsPerBlock = 64;

    hipLaunchKernelGGL(gemv_fim_64cu_64th_fp16, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       (uint8_t*)fim_base /* ctr  base */, (uint8_t*)fim_base + 0x100000 /* weight */,
                       (uint8_t*)fim_base + 0x200000, /* fim hw output */
                       (uint8_t*)input, (uint8_t*)output, 1, n_in_tile, n_out_tile, out_size);
    hipStreamSynchronize(NULL);

    hipFree(input);
    hipFree(output);
}
