#include <stdio.h>
#include <iostream>
#include <string>
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "pim_crf_gen_api.h"

#define SLT_TEST 1

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
            //            PrintHalf(out);
            return 1;
        }
    }
    return 0;
}

__device__ inline void R_CMD(volatile uint8_t* addr)
{
    asm volatile("global_load_dwordx4 v[84:87], %0, off, glc, slc\n\t" ::"v"(addr) : "v84", "v85", "v86", "v87");
}

__device__ inline void W_CMD(volatile uint8_t* addr)
{
    asm volatile("global_store_dwordx4 %0, v[80:83], off, glc, slc\n\t" ::"v"(addr) : "v80", "v81", "v82", "v83");
}

__device__ inline void W_CMD_R(volatile uint8_t* addr, volatile uint8_t* src)
{
    ((ulonglong2*)addr)[0] = ((ulonglong2*)src)[0];
}

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

__global__ void fill_test_16th(volatile uint8_t* pim_ctr, volatile uint8_t* pim_data, volatile uint8_t* pim_data2,
                               volatile uint8_t* output, volatile uint8_t* crf_binary, volatile uint8_t* hab_to_pim,
                               volatile uint8_t* pim_to_hab, volatile uint8_t* test_input1,
                               volatile uint8_t* test_input2, volatile uint8_t* test_input3, int chan)
{
    long long int start;
    int num_bg = 4;
    int num_ba = 4;
    int w_idx = hipThreadIdx_x % 2;
    int gidx = hipThreadIdx_x / 2;
    uint64_t offset = w_idx * 0x10;
    uint64_t addr;
    int ch = hipBlockIdx_x + chan;
    //    int ch = hipBlockIdx_x * 2 + 1;
    int park_col = 0;

    /* intialize values of 0~7 cols */
    for (int bg = 0; bg < num_bg; bg++) {
        for (int ba = 0; ba < num_ba; ba++) {
            addr = addr_gen(ch, 0, bg, ba, 0, gidx);
            W_CMD_R(&output[addr + offset], test_input3 + w_idx * 16);
            B_CMD(1);
            B_CMD(0);
        }
    }

    /* intialize values of 0~7 cols */
    for (int bg = 0; bg < num_bg; bg++) {
        for (int ba = 0; ba < num_ba; ba++) {
            addr = addr_gen(ch, 0, bg, ba, 0, gidx);
            W_CMD_R(&pim_data[addr + offset], test_input1 + w_idx * 16);
            W_CMD_R(&pim_data2[addr + offset], test_input2 + w_idx * 16);
            B_CMD(1);
            B_CMD(0);
        }
    }

#if 1
    /* park in */
    addr = addr_gen(ch, 0, gidx / num_ba, gidx % num_ba, 0, 0);
    R_CMD(&pim_data[addr + offset]);
    addr = addr_gen(ch, 0, gidx / num_ba + 2, gidx % num_ba, 0, 0);
    R_CMD(&pim_data[addr + offset]);
    B_CMD(1);
#endif

    if (hipThreadIdx_x < 2) {
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

        /* set crf binary */
        addr = addr_gen(ch, 0, 0, 1, 0x3fff, 0x4);
        W_CMD_R(&pim_ctr[addr + offset], crf_binary + hipThreadIdx_x * 16);
        B_CMD(1);

        /* change HAB mode to HAB_FIM mode */
        addr = addr_gen(ch, 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R(&pim_ctr[addr + offset], hab_to_pim + hipThreadIdx_x * 16);
        B_CMD(1);
    }
    B_CMD(0);

    /* FILL */
    addr = addr_gen(ch, 0, 0, 0, 0, gidx);  // FILL even_bank to grf_A
    R_CMD(&pim_data[addr + offset]);
    B_CMD(1);

    R_CMD(&pim_data2[addr + offset]);
    B_CMD(1);

    W_CMD(&output[addr + offset]);  // NOP, 7
    W_CMD(&output[addr + offset]);  // NOP, 7
    B_CMD(1);
#if 1
    addr = addr_gen(ch, 0, 0, 1, 0, gidx);  // FILL odd_bank to grf_B
    R_CMD(&pim_data[addr + offset]);
    B_CMD(1);

    R_CMD(&pim_data2[addr + offset]);
    B_CMD(1);

    W_CMD(&output[addr + offset]);  // NOP, 7
    W_CMD(&output[addr + offset]);  // NOP, 7
    B_CMD(1);
#endif
    if (hipThreadIdx_x < 2) {
        /* change HAB_FIM mode to HAB mode */
        addr = addr_gen(ch, 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R(&pim_ctr[addr + offset], pim_to_hab + hipThreadIdx_x * 16);
        B_CMD(1);

        /* change HAB mode to SB mode */
        addr = addr_gen(ch, 0, 0, 0, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        addr = addr_gen(ch, 0, 0, 1, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);

        addr = addr_gen(ch, 0, 0, 0, 0, 0);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
    }
    B_CMD(0);

#if 1
    addr = addr_gen(ch, 0, gidx / num_ba, gidx % num_ba, 0, 0);
    R_CMD(&pim_data[addr + offset]);
    addr = addr_gen(ch, 0, gidx / num_ba + 2, gidx % num_ba, 0, 0);
    R_CMD(&pim_data[addr + offset]);
    B_CMD(1);
#endif
}

int main(int argc, char* argv[])
{
    uint64_t pim_base, pim_out, pim_data, pim_data2;
    uint64_t *mode1_d, *mode2_d, *crf_bin_d, *test1_d, *test2_d, *test3_d;
    uint64_t *mode1_h, *mode2_h, *crf_bin_h, *test1_h, *test2_h, *test3_h;
    size_t N = 4;
    size_t Nbytes = N * sizeof(uint64_t);
    static int device = 0;
    if (argc < 4) {
        printf("usage: ./fill_test <start_ch> <num_ch> <num_iter>\n");
        exit(1);
    }
    int start_ch = std::stoi(argv[1]);
    int num_ch = std::stoi(argv[2]);
    int num_iter = std::stoi(argv[3]);

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
    test1_h = (uint64_t*)malloc(Nbytes);
    CHECK(test1_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    test2_h = (uint64_t*)malloc(Nbytes);
    CHECK(test2_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    test3_h = (uint64_t*)malloc(Nbytes);
    CHECK(test3_h == 0 ? hipErrorOutOfMemory : hipSuccess);

    std::vector<PimCommand> FILL_cmds{PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
                                      PimCommand(PimCmdType::NOP, 7), PimCommand(PimCmdType::EXIT, 0)};

    std::vector<PimCommand> ADD_cmds{
        PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
        PimCommand(PimCmdType::ADD, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1),
        PimCommand(PimCmdType::NOP, 7), PimCommand(PimCmdType::NOP, 7), PimCommand(PimCmdType::EXIT, 0)};

    std::vector<PimCommand> FULL_ADD_cmds{
        PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
        PimCommand(PimCmdType::ADD, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1),
        PimCommand(PimCmdType::NOP, 15),
        //        PimCommand(PimCmdType::NOP, 7),
        PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK),
        PimCommand(PimCmdType::ADD, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1),
        PimCommand(PimCmdType::NOP, 15),
        //        PimCommand(PimCmdType::NOP, 7),
        PimCommand(PimCmdType::EXIT, 0)};

    uint32_t crf_buffer[8] = {
        0,
    };
    for (int i = 0; i < FULL_ADD_cmds.size(); i++) {
        uint32_t u32_data_ = FULL_ADD_cmds[i].to_int();
        memcpy(&crf_buffer[i], &u32_data_, sizeof(uint32_t));
        //        std::cout << std::hex << crf_buffer[i] << std::endl ;
    }

    for (int i = 0; i < 8; i++) {
        ((uint32_t*)crf_bin_h)[i] = crf_buffer[i];
        //        std::cout << std::hex << ((uint32_t*)crf_bin_h)[i] << std::endl ;
    }

    mode1_h[0] = 0x0000000000000001;
    mode1_h[1] = 0x0000010000000000;
    mode1_h[2] = 0x0000000000000000;
    mode1_h[3] = 0x0000000000000000;
    mode2_h[0] = 0x0000000000000000;
    mode2_h[1] = 0x0000000000000000;
    mode2_h[2] = 0x0000000000000000;
    mode2_h[3] = 0x0000000000000000;

    half_float::half golden_out[16];
    for (int i = 0; i < 16; i++) {
        ((half_float::half*)test1_h)[i] = half_float::half(i);
        ((half_float::half*)test2_h)[i] = half_float::half(2.0);
        ((half_float::half*)test3_h)[i] = half_float::half(-1.0);
        golden_out[i] = ((half_float::half*)test1_h)[i] + ((half_float::half*)test2_h)[i];
        //        golden_out[i] = ((half_float::half*)test1_h)[i];
    }

    CHECK(hipMalloc(&crf_bin_d, Nbytes));
    CHECK(hipMalloc(&mode1_d, Nbytes));
    CHECK(hipMalloc(&mode2_d, Nbytes));
    CHECK(hipMalloc(&test1_d, Nbytes));
    CHECK(hipMalloc(&test2_d, Nbytes));
    CHECK(hipMalloc(&test3_d, Nbytes));

    CHECK(hipMemcpy(crf_bin_d, crf_bin_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode1_d, mode1_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode2_d, mode2_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(test1_d, test1_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(test2_d, test2_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(test3_d, test3_h, Nbytes, hipMemcpyHostToDevice));

    int out_row = 2;
    pim_data = pim_base + addr_gen(0, 0, 0, 0, 0, 0);
    pim_data2 = pim_base + addr_gen(0, 0, 0, 0, 1, 0);
    pim_out = pim_base + addr_gen(0, 0, 0, 0, out_row, 0);

    int end_result = 0;
    int* idx_result = (int*)calloc(num_iter, sizeof(int));
    int total_fail_cnt = 0;
    for (int i = 0; i < num_iter; i++) {
        const unsigned blocks = num_ch;
        const unsigned threadsPerBlock = 16;
        hipLaunchKernelGGL(fill_test_16th, dim3(blocks), dim3(threadsPerBlock), 0, 0, (uint8_t*)pim_base,
                           (uint8_t*)pim_data, (uint8_t*)pim_data2, (uint8_t*)pim_out, (uint8_t*)crf_bin_d,
                           (uint8_t*)mode1_d, (uint8_t*)mode2_d, (uint8_t*)test1_d, (uint8_t*)test2_d,
                           (uint8_t*)test3_d, start_ch);

        hipDeviceSynchronize();

        uint64_t addr_offset;
        int fail_cnt = 0;
        int ch_result[64] = {
            0,
        };

        for (int chan = start_ch; chan < (start_ch + num_ch); chan++) {
            for (int bg = 0; bg < 4; bg++) {
                for (int ba = 0; ba < 4; ba++) {
                    for (int col = 0; col < 8; col++) {
                        addr_offset = addr_gen(chan, 0, bg, ba, out_row, col);
                        if (compare_out((uint64_t*)golden_out, (uint64_t*)((uint8_t*)pim_base + addr_offset))) {
                            printf("fail ch: %d bg: %d ba: %d row: %d col: %d\n", chan, bg, ba, out_row, col);
                            PrintHalf((uint64_t*)((uint8_t*)pim_base + addr_offset));
                            ch_result[chan] = 1;
                            fail_cnt++;
                        }
                    }
                }
            }
        }

        if (fail_cnt) {
            printf("idx: %d golden_output\n", i);
            PrintHalf((uint64_t*)golden_out);
            printf("failed ch: ");
            for (int i = start_ch; i < start_ch + num_ch; i++) {
                if (ch_result[i]) {
                    printf("%d ", i);
                }
            }
            printf("\n");
            printf("fail_cnt: %d\n", fail_cnt);
            total_fail_cnt += fail_cnt;
            idx_result[i] = 1;
        } else {
            //            printf("pass\n");
        }
    }

    if (total_fail_cnt) {
        printf("FAIL\n");
        printf("fail_idx: ");
        for (int idx = 0; idx < num_iter; idx++) {
            if (idx_result[idx]) {
                printf("%d ", idx);
            }
        }
        printf("\n");
    } else {
        printf("PASS\n");
    }

    free(mode1_h);
    free(mode2_h);
    free(crf_bin_h);
    free(test1_h);
    free(test2_h);
    free(test3_h);
    free(idx_result);

    hipFree(mode1_d);
    hipFree(mode2_d);
    hipFree(crf_bin_d);
    hipFree(test1_d);
    hipFree(test2_d);
    hipFree(test3_d);
}
