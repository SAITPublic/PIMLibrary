#include <stdio.h>
#include <iostream>
#include <string>
#include "half.hpp"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

using namespace half_float::literal;

#define SLT_TEST 1
//#define CHANNEL 3
#define NUM_ITER 50

#if SLT_TEST
#define TARGET_MASK (0xFFFFFFFFFFFFFFFF)
#else
#define TARGET_MASK (0x1FFFFFFFF)
#endif

extern "C" uint64_t fmm_map_pim(uint32_t, uint32_t, uint64_t);

#define CHECK(cmd)                                                                                              \
    {                                                                                                           \
        hipError_t error = cmd;                                                                                 \
        if (error != hipSuccess) {                                                                              \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    }

__host__ void PrintHalf(uint64_t* data)
{
    for (int i = 0; i < 16; i++) {
        half_float::half x = ((half_float::half*)data)[i];
        printf("%f ", float(x));
    }
    printf("\n");
}

__device__ inline void R_CMD(volatile uint8_t* addr)
{
    asm volatile("global_load_dwordx4 v[24:27], %0, off, glc, slc\n\t" ::"v"(addr) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD(volatile uint8_t* addr)
{
    asm volatile("global_store_dwordx4 %0, v[24:27], off, glc, slc\n\t" ::"v"(addr) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD_R(volatile uint8_t* addr, volatile uint8_t* src) { ((int4*)addr)[0] = ((int4*)src)[0]; }

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

__host__ __device__ inline unsigned int mask_by_bit(unsigned int value, int start, int end)
{
    int length = start - end + 1;
    value = value >> end;
    return value & ((1 << length) - 1);
}

__host__ __device__ uint64_t addr_gen(unsigned int ch, unsigned int rank, unsigned int bg, unsigned int ba,
                                      unsigned int row, unsigned int col)
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

#if 0
    uint64_t mask = TARGET_MASK;
    addr &= mask;
#endif

    return addr;
}

__global__ void grf_test(volatile uint8_t* pim_ctr, volatile uint8_t* pim_data, volatile uint8_t* output,
                         volatile uint8_t* crf_binary, volatile uint8_t* hab_to_pim, volatile uint8_t* pim_to_hab,
                         volatile uint8_t* test_input1, volatile uint8_t* test_input2, int chan, int row)
{
    uint64_t offset = hipThreadIdx_x * 0x10;
    uint64_t addr;
    int ch = chan + hipThreadIdx_x / 2;
    int park_row = 0x3000;
    int park_col = 0;

    long long int start;

    for (int grf_row = 0; grf_row < NUM_ITER; grf_row++) {
        addr = addr_gen(ch, 0, 0, 0, grf_row, 0);
        W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
#if 0
    addr = addr_gen(ch, 0, 0, 0, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    addr = addr_gen(ch, 0, 0, 1, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    addr = addr_gen(ch, 0, 0, 2, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    addr = addr_gen(ch, 0, 1, 0, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    addr = addr_gen(ch, 0, 1, 2, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    addr = addr_gen(ch, 0, 2, 0, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    addr = addr_gen(ch, 0, 2, 2, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    addr = addr_gen(ch, 0, 3, 0, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    addr = addr_gen(ch, 0, 3, 2, grf_row, 0);
    W_CMD_R(&output[addr + offset], test_input1 + hipThreadIdx_x * 16);
    B_CMD(1);
#endif
#if 0
    /* park in */
    addr = addr_gen(ch, 0, 0, 0, 0x20, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 0, 1, 0x40, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 0, 2, 0x60, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 0, 3, 0x80, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 1, 0, 0xa0, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 1, 1, 0xc0, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 1, 2, 0xe0, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 1, 3, 0x100, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 2, 0, 0x120, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 2, 1, 0x140, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 2, 2, 0x160, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 2, 3, 0x180, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 3, 0, 0x1a0, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 3, 1, 0x1c0, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 3, 2, 0x1e0, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 3, 3, 0x200, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
#endif
#if 1

        addr = addr_gen(ch, 0, 2, 0, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);

        addr = addr_gen(ch, 0, 2, 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);

        addr = addr_gen(ch, 0, 0, 0, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);

        addr = addr_gen(ch, 0, 0, 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);

#endif
#if 1
        addr = addr_gen(ch, 0, 0, 0, 0x3fff, 0x8);
        W_CMD_R(&pim_ctr[addr + offset], test_input2 + hipThreadIdx_x * 16);  // write to grf_A
        B_CMD(1);

        addr = addr_gen(ch, 0, 0, 0, grf_row, 0);
        W_CMD(&output[addr + offset]);  // NOP
        B_CMD(1);
#endif
#if 1
        /* change HAB mode to SB mode */
        addr = addr_gen(ch, 0, 0, 0, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen(ch, 0, 0, 1, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
        addr = addr_gen(ch, 0, 0, 0, 0x3020, park_col);
        R_CMD(&pim_ctr[addr + offset]);
#if 0
    /* park out */
    addr = addr_gen(ch, 0, 0, 0, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 0, 1, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 0, 2, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 0, 3, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 1, 0, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 1, 1, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 1, 2, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 1, 3, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 2, 0, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 2, 1, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 2, 2, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 2, 3, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 3, 0, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 3, 1, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 3, 2, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    addr = addr_gen(ch, 0, 3, 3, park_row, park_col);
    R_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
#endif
    }
}

int main(int argc, char* argv[])
{
    uint64_t pim_base, pim_out;
    uint64_t *mode1_d, *mode2_d, *crf_bin_d, *test1_d, *test2_d;
    uint64_t *mode1_h, *mode2_h, *crf_bin_h, *test1_h, *test2_h;
    uint64_t* output;
    size_t N = 4;
    size_t Nbytes = N * sizeof(uint64_t);
    static int device = 0;
    int ch = std::stoi(argv[1]);

    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s global mem size: %zu\n", props.name, props.totalGlobalMem);

    // Get GPU ID
    FILE* fd;
    char path[256];
    uint32_t gpu_id;

    snprintf(path, 256, "/sys/devices/virtual/kfd/kfd/topology/nodes/2/gpu_id");
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
#if SLT_TEST
    uint64_t bsize = 17179869184;  // 16 * 1024 * 1024 * 1024;
#else
    uint64_t bsize = 8589934592;  // 8 * 1024 * 1024 * 1024;
#endif
    pim_base = fmm_map_pim(2, gpu_id, bsize);
    std::cout << std::hex << "pimBaseAddr = " << pim_base << std::endl;

    crf_bin_h = (uint64_t*)malloc(Nbytes);
    CHECK(crf_bin_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    mode1_h = (uint64_t*)malloc(Nbytes);
    CHECK(mode1_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    mode2_h = (uint64_t*)malloc(Nbytes);
    CHECK(mode2_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    test1_h = (uint64_t*)malloc(Nbytes);
    CHECK(test1_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    test2_h = (uint64_t*)malloc(Nbytes);
    CHECK(test2_h == 0 ? hipErrorOutOfMemory : hipSuccess);

    mode1_h[0] = 0x0000000000000001;
    mode1_h[1] = 0x0000010000000000;
    mode1_h[2] = 0x0000000000000000;
    mode1_h[3] = 0x0000000000000000;
    mode2_h[0] = 0x0000000000000000;
    mode2_h[1] = 0x0000000000000000;
    mode2_h[2] = 0x0000000000000000;
    mode2_h[3] = 0x0000000000000000;
    test1_h[0] = 0x0000000000000001;
    test1_h[1] = 0x0000000000000001;
    test1_h[2] = 0x0000000000000001;
    test1_h[3] = 0x0000000000000001;
    test2_h[0] = 0x000000000000000a;
    test2_h[1] = 0x000000000000000b;
    test2_h[2] = 0x000000000000000c;
    test2_h[3] = 0x000000000000000d;

    CHECK(hipMalloc(&crf_bin_d, Nbytes));
    CHECK(hipMalloc(&mode1_d, Nbytes));
    CHECK(hipMalloc(&mode2_d, Nbytes));
    CHECK(hipMalloc(&test1_d, Nbytes));
    CHECK(hipMalloc(&test2_d, Nbytes));

    CHECK(hipMemcpy(crf_bin_d, crf_bin_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode1_d, mode1_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode2_d, mode2_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(test1_d, test1_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(test2_d, test2_h, Nbytes, hipMemcpyHostToDevice));

    int out_row = 0x40;
    pim_out = pim_base + addr_gen(0, 0, 0, 0, out_row, 0);

    const unsigned blocks = 1;
    const unsigned threadsPerBlock = 4;
    int failed = 0;
    int fail_cnt = 0;

    uint64_t addr_offset;
    int num_iter = 1;
    for (int i = 0; i < num_iter; i++) {
        hipDeviceSynchronize();
        hipLaunchKernelGGL(grf_test, dim3(blocks), dim3(threadsPerBlock), 0, 0, (uint8_t*)pim_base, (uint8_t*)pim_base,
                           (uint8_t*)pim_out, (uint8_t*)crf_bin_d, (uint8_t*)mode1_d, (uint8_t*)mode2_d,
                           (uint8_t*)test1_d, (uint8_t*)test2_d, ch, i);
        hipDeviceSynchronize();
    }

    for (int i = 0; i < NUM_ITER; i++) {
        failed = 0;
        //        for (int ba=0; ba<4; ba+=2) {
        //            for (int bg=0; bg<4; bg++) {
        //                addr_offset = addr_gen(ch, 0, bg, ba, out_row + i, 0);
        addr_offset = addr_gen(ch, 0, 0, 0, out_row + i, 0);
        output = (uint64_t*)((uint8_t*)pim_base + addr_offset);
        //                printf("%#018lx %#018lx %#018lx %#018lx\n", output[3], output[2], output[1],
        //                output[0]);
        if (output[0] != test2_h[0] || output[1] != test2_h[1] || output[2] != test2_h[2] || output[3] != test2_h[3]) {
            printf("ch: %d failed idx: %d\n", ch, i);
            printf("%#018lx %#018lx %#018lx %#018lx\n", output[3], output[2], output[1], output[0]);
            failed = 1;
        }
        addr_offset = addr_gen(ch + 1, 0, 0, 0, out_row + i, 0);
        output = (uint64_t*)((uint8_t*)pim_base + addr_offset);
        //                printf("%#018lx %#018lx %#018lx %#018lx\n", output[3], output[2], output[1],
        //                output[0]);
        if (output[0] != test2_h[0] || output[1] != test2_h[1] || output[2] != test2_h[2] || output[3] != test2_h[3]) {
            printf("ch: %d failed idx: %d\n", ch + 1, i);
            printf("%#018lx %#018lx %#018lx %#018lx\n", output[3], output[2], output[1], output[0]);
            failed = 1;
        }
        // else {
        //    printf("success idx: %d\n", i);
        //    printf("%#018lx %#018lx %#018lx %#018lx\n", output[3], output[2], output[1], output[0]);
        // }
        //            }
        //        }

        if (failed) fail_cnt++;
    }

    printf("ch: %d fail_cnt: %d\n", ch, fail_cnt);

    free(mode1_h);
    free(mode2_h);
    free(crf_bin_h);
    free(test1_h);
    free(test2_h);

    hipFree(mode1_d);
    hipFree(mode2_d);
    hipFree(crf_bin_d);
    hipFree(test1_d);
    hipFree(test2_d);
}
