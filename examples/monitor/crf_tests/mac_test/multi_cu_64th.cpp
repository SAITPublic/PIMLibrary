#include <stdio.h>
#include <iostream>
#include <string>
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "pim_crf_gen_api.h"

#define SLT_TEST 1
#define BLOCKS 1

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

__host__ int compare_out(uint64_t* golden, uint64_t* out)
{
    for (int i = 0; i < 16; i++) {
        if (((half_float::half*)golden)[i] != ((half_float::half*)out)[i]) {
            PrintHalf(out);
            return 1;
        }
    }
    return 0;
}

__device__ inline void R_CMD(volatile uint8_t* addr)
{
    asm volatile("global_load_dwordx4 v[20:23], %0, off, glc, slc\n\t" ::"v"(addr) : "v20", "v21", "v22", "v23");
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

    uint64_t mask = TARGET_MASK;
    addr &= mask;

    return addr;
}

__global__ void mac_test_64th(volatile uint8_t* pim_ctr, volatile uint8_t* pim_data, volatile uint8_t* output,
                              volatile uint8_t* crf_binary, volatile uint8_t* hab_to_pim, volatile uint8_t* pim_to_hab,
                              volatile uint8_t* test_input1, volatile uint8_t* test_input2, int chan)
{
    int num_bg = 4;
    int num_ba = 4;
    int w_idx = hipThreadIdx_x % 2;
    int gidx = hipThreadIdx_x / 2;
    uint64_t offset = w_idx * 0x10;
    uint64_t addr;
    int ch = hipBlockIdx_x + chan;

    if (hipThreadIdx_x < 16) {
        /* intialize values of 0~7 cols */
        for (int bg = 0; bg < 4; bg++) {
            for (int ba = 0; ba < 4; ba++) {
                addr = addr_gen(ch, 0, bg, ba, 0, gidx);
                W_CMD_R(&pim_data[addr + offset], test_input1 + gidx * 16);
            }
        }
    }
    B_CMD(0);

    /* park */
    if (hipThreadIdx_x < 32) {
        addr = addr_gen(ch, 0, gidx / num_ba, gidx % num_ba, 0, 0);
        R_CMD(&pim_ctr[addr + offset]);
    }
    B_CMD(0);

    if (hipThreadIdx_x < 2) {
#if 1
        /* change SB mode to HAB mode */
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
        /* set crf binary */
        addr = addr_gen(ch, 0, 0, 1, 0x3fff, 0x4);
        W_CMD_R(&pim_ctr[addr + offset], crf_binary + hipThreadIdx_x * 16);
        B_CMD(1);
#endif
    }
    B_CMD(0);

    if (hipThreadIdx_x < 16) {
#if 1
        /* change HAB mode to HAB_FIM mode */
        // FIX : Since there are 16 threads, the mode change is executed 8 times.
        // We need to test it with two threads.
        addr = addr_gen(ch, 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R(&pim_ctr[addr + offset], hab_to_pim + w_idx * 16);
        B_CMD(1);
#endif

#if 1
        /* write grf_A from WRIO */
        addr = addr_gen(ch, 0, 0, 1, 0x3fff, 0x8 + gidx);
        W_CMD_R(&pim_ctr[addr + offset], test_input2 + w_idx * 16);
        B_CMD(1);

        addr = addr_gen(ch, 0, 0, 0, 0, gidx);
        R_CMD(&pim_data[addr + offset]);
        B_CMD(1);

        // pipeline delay
        // FIX : If alu is in operation, NOP should be added.
        addr = addr_gen(ch, 0, 0, 1, 0, gidx);
        W_CMD(&output[addr + offset]);
        B_CMD(1);
#endif

#if 1
        addr = addr_gen(ch, 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R(&pim_ctr[addr + offset], pim_to_hab + w_idx * 16);
        B_CMD(1);
#endif
    }
    B_CMD(0);

    if (hipThreadIdx_x < 2) {
#if 1
        /* change HAB mode to SB mode */
        addr = addr_gen(ch, 0, 0, 0, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        addr = addr_gen(ch, 0, 0, 1, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
    }
    B_CMD(0);
#if 1
    /* park */
    if (hipThreadIdx_x < 32) {
        addr = addr_gen(ch, 0, gidx / num_ba, gidx % num_ba, 0, 0);
        R_CMD(&pim_ctr[addr + offset]);
    }
    B_CMD(0);

#endif
}

int main(int argc, char* argv[])
{
    uint64_t pim_base, pim_out, pim_data;
    uint64_t *mode1_d, *mode2_d, *crf_bin_d, *test1_d, *test2_d, *test3_d;
    uint64_t *mode1_h, *mode2_h, *crf_bin_h, *test1_h, *test2_h, *test3_h;
    uint64_t* output;
    size_t N = 4;
    size_t Nbytes = N * sizeof(uint64_t);
    static int device = 0;
    if (argc < 2) {
        printf("usage: ./fill_test <ch>\n");
        exit(1);
    }
    int ch = std::stoi(argv[1]);

    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s global mem size: %zu\n", props.name, props.totalGlobalMem);

    // Get GPU ID
    FILE* fd;
    char path[256];
    uint32_t gpu_id;
    int node_id = 2;

    snprintf(path, 256, "/sys/devices/virtual/kfd/kfd/topology/nodes/%d/gpu_id", node_id);
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
    pim_base = fmm_map_pim(node_id, gpu_id, bsize);
    std::cout << std::hex << "pimBaseAddr = " << pim_base << std::endl;

    crf_bin_h = (uint64_t*)malloc(Nbytes);
    CHECK(crf_bin_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    mode1_h = (uint64_t*)malloc(Nbytes);
    CHECK(mode1_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    mode2_h = (uint64_t*)malloc(Nbytes);
    CHECK(mode2_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    test1_h = (uint64_t*)malloc(Nbytes * 8);
    CHECK(test1_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    test2_h = (uint64_t*)malloc(Nbytes * 8);
    CHECK(test2_h == 0 ? hipErrorOutOfMemory : hipSuccess);

    std::vector<PimCommand> MAC_cmds{
        PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0),
        PimCommand(PimCmdType::NOP, 7), PimCommand(PimCmdType::EXIT, 0)};

    uint32_t crf_buffer[8] = {
        0,
    };
    for (int i = 0; i < MAC_cmds.size(); i++) {
        uint32_t u32_data_ = MAC_cmds[i].to_int();
        memcpy(&crf_buffer[i], &u32_data_, sizeof(uint32_t));
        //        std::cout << std::hex << crf_buffer[i] << std::endl ;
    }

    for (int i = 0; i < 8; i++) {
        ((uint32_t*)crf_bin_h)[i] = crf_buffer[i];
        //        std::cout << std::hex << ((uint32_t*)crf_bin_h)[i] << std::endl ;
    }

    mode1_h[0] = 0x0000000000000001;
    mode1_h[1] = 0x0000000000000000;
    mode1_h[2] = 0x0000010000000000;
    mode1_h[3] = 0x0000000000000000;
    mode2_h[0] = 0x0000000000000000;
    mode2_h[1] = 0x0000000000000000;
    mode2_h[2] = 0x0000000000000000;
    mode2_h[3] = 0x0000000000000000;

    half_float::half golden_out[128];
    for (int i = 0; i < 128; i++) {
        ((half_float::half*)test1_h)[i] = half_float::half(1.2);
        ((half_float::half*)test2_h)[i] = half_float::half(3.0);
        golden_out[i] = ((half_float::half*)test1_h)[i] * ((half_float::half*)test2_h)[i];
    }

    CHECK(hipMalloc(&crf_bin_d, Nbytes));
    CHECK(hipMalloc(&mode1_d, Nbytes));
    CHECK(hipMalloc(&mode2_d, Nbytes));
    CHECK(hipMalloc(&test1_d, Nbytes));
    CHECK(hipMalloc(&test2_d, Nbytes));
    //    CHECK(hipMalloc(&test3_d, Nbytes));

    CHECK(hipMemcpy(crf_bin_d, crf_bin_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode1_d, mode1_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode2_d, mode2_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(test1_d, test1_h, 8 * Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(test2_d, test2_h, 8 * Nbytes, hipMemcpyHostToDevice));
    //    CHECK(hipMemcpy(test3_d, test3_h, 8* Nbytes, hipMemcpyHostToDevice));

    int out_row = 2;
    pim_data = pim_base + addr_gen(0, 0, 0, 0, 0, 0);
    pim_out = pim_base + addr_gen(0, 0, 0, 0, out_row, 0);

    const unsigned blocks = BLOCKS;
    const unsigned threadsPerBlock = 64;

    hipLaunchKernelGGL(mac_test_64th, dim3(blocks), dim3(threadsPerBlock), 0, 0, (uint8_t*)pim_base, (uint8_t*)pim_data,
                       (uint8_t*)pim_out, (uint8_t*)crf_bin_d, (uint8_t*)mode1_d, (uint8_t*)mode2_d, (uint8_t*)test1_d,
                       (uint8_t*)test2_d, ch);

    hipDeviceSynchronize();

    uint64_t addr_offset;
    int failed;
    int fail_cnt = 0;
    int ch_result[64] = {
        0,
    };
    int total_cnt = 0;

    for (int chan = ch; chan < (ch + BLOCKS); chan++) {
        //    for (int chan = 0; chan < 64; chan+=2) {
        for (int bg = 0; bg < 4; bg++) {
            for (int ba = 1; ba < 4; ba += 2) {
                failed = 0;
                printf("ch: %d bg: %d ba: %d row: %d\n", chan, bg, ba, out_row);
                for (int col = 0; col < 8; col++) {
                    addr_offset = addr_gen(chan, 0, bg, ba, out_row, col);
                    output = (uint64_t*)((uint8_t*)pim_base + addr_offset);
                    if (failed) PrintHalf(output);
                    // printf("%#018lx %#018lx %#018lx %#018lx\n", output[3], output[2], output[1], output[0]);
                }
                if (failed) {
                    ch_result[chan] = 1;
                    fail_cnt++;
                    printf("fail\n");
                } else {
                    printf("pass\n");
                }
            }
        }
    }

    printf("golden_output\n");
    PrintHalf((uint64_t*)golden_out);
    printf("failed: ");
    for (int i = ch; i < ch + blocks; i++) {
        //    for (int i = 0; i < 64; i+=2) {
        if (ch_result[i]) {
            printf("%d ", i);
        }
    }
    printf("\n");
    printf("fail_cnt: %d\n", fail_cnt);

    free(mode1_h);
    free(mode2_h);
    free(crf_bin_h);
    free(test1_h);
    free(test2_h);
    //    free(test3_h);

    hipFree(mode1_d);
    hipFree(mode2_d);
    hipFree(crf_bin_d);
    hipFree(test1_d);
    hipFree(test2_d);
    hipFree(test3_d);
}
