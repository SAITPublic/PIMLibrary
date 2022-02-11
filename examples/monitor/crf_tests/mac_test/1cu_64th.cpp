#include <stdio.h>
#include <time.h>
#include <iostream>
#include <random>
#include <string>
#include "half.hpp"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "pim_crf_gen_api.h"

#define SLT_TEST 1
#define BLOCKS 1
#define NUM_ITER 1

#if SLT_TEST
#define TARGET_MASK (0xFFFFFFFFFFFFFFFF)
#else
#define TARGET_MASK (0x1FFFFFFFF)
#endif

extern "C" uint64_t fmm_map_pim(uint32_t, uint32_t, uint64_t);

long long getTickCount(void)
{
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (long long)tp.tv_sec * 1000000000 + tp.tv_nsec;
}

double getTickFrequency(void) { return 1e9; }

#define FIM_PROFILE_TICK(name) long long __tick_##name = getTickCount()
#define FIM_PROFILE_TOCK(name)                                                                                     \
    std::cout << #name                                                                                             \
              << " time (ms) : " << ((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000 \
              << std::endl;

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

__host__ void PrintHalf(half_float::half* data)
{
    for (int i = 0; i < 16; i++) {
        printf("%f ", float(data[i]));
    }
    printf("\n");
}

__host__ int compare_out(uint64_t* golden, uint64_t* out)
{
    for (int i = 0; i < 16; i++) {
        if (abs(float(((half_float::half*)golden)[i] - ((half_float::half*)out)[i])) > 0.1) {
            std::cout << " out : ";
            PrintHalf(out);
            std::cout << "golden : ";
            PrintHalf(golden);
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
                              volatile uint8_t* temp_output, volatile uint8_t* crf_binary, volatile uint8_t* hab_to_pim,
                              volatile uint8_t* pim_to_hab, volatile uint8_t* weight, volatile uint8_t* input, int chan,
                              int num_input_tile)
{
    int num_bg = 4;
    int num_ba = 4;
    int w_idx = hipThreadIdx_x % 2;
    int gidx = hipThreadIdx_x / 2;
    uint64_t offset = w_idx * 0x10;
    uint64_t addr;
    int ch = hipBlockIdx_x + chan;

    /* park */
    if (hipThreadIdx_x < 32) {
        addr = addr_gen(ch, 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
        R_CMD(&pim_ctr[addr + offset]);
    }
    B_CMD(0);

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
    }
    B_CMD(0);

    int even_row, odd_row;
    if (hipThreadIdx_x < 16) {
        /* change HAB mode to HAB_FIM mode */
        // FIX : Since there are 16 threads, the mode change is executed 8 times.
        // We need to test it with two threads.
        for (int b = 0; b < 1 /* batch_dim*/; b++) {
            for (int j = 0; j < 1 /* num_out_tile*/; j++) {
                addr = addr_gen(ch, 0, 0, 0, 0x3fff, 0x0);
                W_CMD_R(&pim_ctr[addr + offset], hab_to_pim + w_idx * 16);
                B_CMD(1);

                for (int i_idx = 0; i_idx < num_input_tile /*n_compute_tile*/; i_idx += 2) {
                    /* write grf_A from WRIO */
                    addr = addr_gen(ch, 0, 0, 1, 0x3fff, 0x8 + gidx);
                    W_CMD_R(&pim_ctr[addr + offset], input + ((i_idx * 8 + gidx) * 32 + (w_idx * 16)));
                    B_CMD(1);

                    even_row = (i_idx >> 1) << 1;
                    odd_row = even_row + 1;

                    addr = addr_gen(ch, 0, 0, 0, even_row, gidx);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 0, even_row, gidx + 8);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 0, even_row, gidx + 16);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 0, even_row, gidx + 24);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 0, odd_row, gidx);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 0, odd_row, gidx + 8);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 0, odd_row, gidx + 16);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 0, odd_row, gidx + 24);
                    R_CMD(&pim_data[addr + offset]);
                    B_CMD(1);
                }

                for (int i_idx = 1; i_idx < num_input_tile /*n_compute_tile*/; i_idx += 2) {
                    addr = addr_gen(ch, 0, 0, 0, 0x3fff, 0x8 + gidx);
                    W_CMD_R(&pim_ctr[addr + offset], input + (i_idx * 8 + gidx) * 32 + (w_idx * 16));
                    B_CMD(1);

                    even_row = (i_idx >> 1) << 1;
                    odd_row = even_row + 1;

                    addr = addr_gen(ch, 0, 0, 1, even_row, gidx);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 1, even_row, gidx + 8);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 1, even_row, gidx + 16);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 1, even_row, gidx + 24);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 1, odd_row, gidx);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 1, odd_row, gidx + 8);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 1, odd_row, gidx + 16);
                    R_CMD(&pim_data[addr + offset]);

                    addr = addr_gen(ch, 0, 0, 1, odd_row, gidx + 24);
                    R_CMD(&pim_data[addr + offset]);
                    B_CMD(1);
                }

                // pipeline delay
                // FIX : If alu is in operation, NOP should be added.
                addr = addr_gen(ch, 0, 0, 1, 0, gidx);
                W_CMD(&temp_output[addr + offset]);
                B_CMD(1);

                addr = addr_gen(ch, 0, 0, 0, 0x3fff, 0x0);
                W_CMD_R(&pim_ctr[addr + offset], pim_to_hab + w_idx * 16);
                B_CMD(1);
            }
        }
    }

    B_CMD(0);

    if (hipThreadIdx_x < 2) {
        /* change HAB mode to SB mode */
        addr = addr_gen(ch, 0, 0, 0, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        addr = addr_gen(ch, 0, 0, 1, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
    }
    B_CMD(0);

    /* park */
    if (hipThreadIdx_x < 32) {
        addr = addr_gen(ch, 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
        R_CMD(&pim_ctr[addr + offset]);
    }
    B_CMD(0);

#if 1
    /* TODO: verify reduce sum in Target Mode */
    int bg = hipThreadIdx_x >> 4;
    int ba = (((hipThreadIdx_x >> 3) % 2) << 1) + 1;
    int out_idx = (hipBlockIdx_x << 6) + hipThreadIdx_x;
    int row;
    int col;

    half t_output;

    for (int i = 0; i < 1 /*batch_dim * num_out_tile*/; i++) {
        if (out_idx < 64 /*output_dim*/) {
            t_output = 0;
            row = 0;
            col = hipThreadIdx_x % 8 + ((i % 4) << 3);
            addr = addr_gen(ch, 0, bg, ba, row, col);
            for (int j = 0; j < 16; j++) {
                t_output += ((half*)temp_output)[(addr >> 1) + j];
            }
            ((half*)output)[out_idx] = t_output;
        }
    }
#endif
}

int main(int argc, char* argv[])
{
    uint64_t pim_base, pim_out, pim_data, pim_temp_output;
    uint64_t *mode1_d, *mode2_d, *crf_bin_d;
    uint64_t *mode1_h, *mode2_h, *crf_bin_h;
    uint64_t* output;

    uint64_t *input_host, *weight_host;
    uint64_t *input_device, *weight_device;

    size_t N = 4;
    size_t Nbytes = N * sizeof(uint64_t);
    static int device = 0;

    int num_input_tile = 2;
    int num_fp16_per_grf = 16;
    int num_grf_A = 8;
    int num_grf_B = 8;
    int num_pimblock = 8;

    int num_input = num_grf_A * num_input_tile * num_fp16_per_grf;
    int num_output = num_grf_B * num_pimblock;
    int num_weight = num_input * num_output;

    std::cout << " num_input : " << num_input << std::endl;
    std::cout << " num_output : " << num_output << std::endl;

    if (argc < 2) {
        printf("usage: ./fill_test <ch>\n");
        exit(1);
    }
    int ch = std::stoi(argv[1]);

    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s global mem size: %zu\n", props.name, props.totalGlobalMem);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;

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

    input_host = (uint64_t*)malloc(num_input * 2);
    CHECK(input_host == 0 ? hipErrorOutOfMemory : hipSuccess);
    weight_host = (uint64_t*)malloc(num_weight * 2);
    CHECK(weight_host == 0 ? hipErrorOutOfMemory : hipSuccess);

    std::vector<PimCommand> MAC_cmds{
        PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0),
        PimCommand(PimCmdType::JUMP, 7, 2),
        PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::ODD_BANK, 1, 0, 0, 0),
        PimCommand(PimCmdType::JUMP, 7, 2),
        PimCommand(PimCmdType::NOP, 7),
        PimCommand(PimCmdType::EXIT, 0)};

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
        std::cout << std::hex << ((uint32_t*)crf_bin_h)[i] << std::endl;
    }

    mode1_h[0] = 0x0000000000000001;
    mode1_h[1] = 0x0000000000000000;
    mode1_h[2] = 0x0000010000000000;
    mode1_h[3] = 0x0000000000000000;
    mode2_h[0] = 0x0000000000000000;
    mode2_h[1] = 0x0000000000000000;
    mode2_h[2] = 0x0000000000000000;
    mode2_h[3] = 0x0000000000000000;

    half_float::half golden_out[1024];
    half_float::half reduced_result[64];
    half_float::half reduced_result_16[64];
    for (int i = 0; i < 1024; i++) {
        golden_out[i] = 0.0;
    }
    for (int i = 0; i < 64; i++) {
        reduced_result[i] = 0.0;
        reduced_result_16[i] = 0.0;
    }

    for (int i = 0; i < num_input; i++) {
        ((half_float::half*)input_host)[i] = half_float::half(dis(gen));
    }

    for (int i = 0; i < num_weight; i++) {
        ((half_float::half*)weight_host)[i] = half_float::half(dis(gen));
    }

    for (int o = 0; o < num_output; o++) {
        for (int i = 0; i < num_input; i += 16) {
            for (int f_idx = 0; f_idx < 16; f_idx++) {
                // half_float::half a =
                // golden_out[o * 16 + f_idx] += ((half_float::half*)input_host)[i + f_idx] *
                // ((half_float::half*)weight_host)[o * num_input +  (i + f_idx)];
                golden_out[o * 16 + f_idx] += ((half_float::half*)input_host)[i + f_idx] *
                                              ((half_float::half*)weight_host)[o * num_input + (i + f_idx)];
            }
        }
    }

    for (int o = 0; o < num_output; o++) {
        for (int t = 0; t < 16; t++) {
            reduced_result_16[o] += golden_out[o * 16 + t];
        }

        for (int i = 0; i < num_input; i++) {
            reduced_result[o] +=
                ((half_float::half*)input_host)[i] * ((half_float::half*)weight_host)[o * num_input + i];
        }
    }

    // for (int o=0; o<num_output; o++) {
    //     std::cout << std::dec <<o << " result : " << (float)reduced_result[o]  << " : " <<
    //     (float)reduced_result_16[o] <<std::endl;
    // }

    int bg = 0;
    int bank = 0;
    uint32_t col = 0;
    uint32_t row = 0;
    uint32_t even_col = 0;
    uint32_t even_row = 0;
    uint32_t odd_col = 0;
    uint32_t odd_row = 0;
    uint64_t addr;

    char* src_data = (char*)weight_host;
    int in_cnt = 8 * num_input_tile;
    for (int x = 0; x < in_cnt; x += 8) {
        if ((x / 8) % 2 == 0) {
            for (int tiled_y = 0; tiled_y < 64; tiled_y += 8) {
                col = even_col;
                row = even_row;

                for (int grfb_idx = 0; grfb_idx < 8; grfb_idx++) {
                    for (int grfa_idx = 0; grfa_idx < 8; grfa_idx++) {
                        addr = addr_gen(ch, 0, bg, bank, row, col);
                        int d_idx = (tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
                        // std::cout <<" d_idx : " << std::dec<< d_idx << std::endl;
                        memcpy((char*)pim_base + addr, src_data + d_idx * 32, 32);
                        col++;
                        if (col >= 32) {
                            row++;
                            col = 0;
                        }
                    }
                }

                bank += 2;

                if (bank >= 4) {
                    bg++;
                    bank = 0;
                }

                if (bg >= 4) {
                    bg = 0;
                    even_row = row;
                    even_col = col;
                }
            }
        }

        if ((x / 8) % 2 == 1) {
            for (int tiled_y = 0; tiled_y < 64; tiled_y += 8) {
                col = odd_col;
                row = odd_row;

                for (int grfb_idx = 0; grfb_idx < 8; grfb_idx++) {
                    for (int grfa_idx = 0; grfa_idx < 8; grfa_idx++) {
                        addr = addr_gen(ch, 0, bg, bank + 1, row, col);
                        int d_idx = (tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
                        // std::cout <<" d_idx2 : " << std::dec<< d_idx << std::endl;
                        memcpy((char*)pim_base + addr, src_data + d_idx * 32, 32);
                        col++;
                        if (col >= 32) {
                            row++;
                            col = 0;
                        }
                    }
                }

                bank += 2;

                if (bank >= 4) {
                    bg++;
                    bank = 0;
                }

                if (bg >= 4) {
                    bg = 0;
                    odd_row = row;
                    odd_col = col;
                }
            }
        }
    }

    CHECK(hipMalloc(&crf_bin_d, Nbytes));
    CHECK(hipMalloc(&mode1_d, Nbytes));
    CHECK(hipMalloc(&mode2_d, Nbytes));
    CHECK(hipMalloc(&input_device, num_input * 2));
    CHECK(hipMalloc(&weight_device, num_weight * 2));
    //    CHECK(hipMalloc(&test3_d, Nbytes));

    CHECK(hipMemcpy(crf_bin_d, crf_bin_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode1_d, mode1_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode2_d, mode2_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(input_device, input_host, num_input * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(weight_device, weight_host, num_weight * 2, hipMemcpyHostToDevice));
    //    CHECK(hipMemcpy(test3_d, test3_h, 8* Nbytes, hipMemcpyHostToDevice));

    int out_row = 100;
    int temp_out_row = 200;
    pim_data = pim_base + addr_gen(0, 0, 0, 0, 0, 0);
    pim_out = pim_base + addr_gen(0, 0, 0, 0, out_row, 0);
    pim_temp_output = pim_base + addr_gen(0, 0, 0, 0, temp_out_row, 0);

    const unsigned blocks = BLOCKS;
    const unsigned threadsPerBlock = 64;

    int total_fail_cnt = 0;
    for (int i = 0; i < NUM_ITER; i++) {
        hipLaunchKernelGGL(mac_test_64th, dim3(blocks), dim3(threadsPerBlock), 0, 0, (uint8_t*)pim_base,
                           (uint8_t*)pim_data, (uint8_t*)pim_out, (uint8_t*)pim_temp_output, (uint8_t*)crf_bin_d,
                           (uint8_t*)mode1_d, (uint8_t*)mode2_d, (uint8_t*)weight_device, (uint8_t*)input_device, ch,
                           num_input_tile);
        hipDeviceSynchronize();
#if 0
    FIM_PROFILE_TICK(GEMV);
    for (int i = 0; i < 1000; i++) {
    hipLaunchKernelGGL(mac_test_64th, dim3(blocks), dim3(threadsPerBlock), 0, 0, (uint8_t*)pim_base, (uint8_t*)pim_data, 
            (uint8_t*)pim_out, (uint8_t*)pim_temp_output, (uint8_t*)crf_bin_d, (uint8_t*)mode1_d, (uint8_t*)mode2_d, (uint8_t*)weight_device,
            (uint8_t*)input_device, ch, num_input_tile);
    }
    hipDeviceSynchronize();
    FIM_PROFILE_TOCK(GEMV);
#endif
        uint64_t addr_offset;
        int failed;
        int fail_cnt = 0;
        int ch_result[64] = {
            0,
        };

#if 1
        int gi = 0;
        for (int chan = ch; chan < (ch + BLOCKS); chan++) {
            for (int bg = 0; bg < 4; bg++) {
                for (int ba = 1; ba < 4; ba += 2) {
                    failed = 0;
                    printf("ch: %d bg: %d ba: %d row: %d\n", chan, bg, ba, temp_out_row);
                    for (int col = 0; col < 8; col++) {
                        std::cout << " col : " << col << " : ";
                        addr_offset = addr_gen(chan, 0, bg, ba, temp_out_row, col);
                        output = (uint64_t*)((uint8_t*)pim_base + addr_offset);
                        failed = compare_out((uint64_t*)golden_out + (gi++) * 4, output);
                        if (failed) {
                            ch_result[chan] = 1;
                            fail_cnt++;
                            printf("fail\n");
                        } else {
                            printf("pass\n");
                        }
                        // if (failed) PrintHalf(output);
                        // printf("[3] %#018lx [2] %#018lx [1] %#018lx [0] %#018lx\n", output[3], output[2],
                        // output[1], output[0]);
                    }
                }
            }
        }

        for (int i = 0; i < num_output; i++) {
            std::cout << " result : " << std::dec << reduced_result_16[i] << " "
                      << float(((half_float::half*)pim_out)[i]) << std::endl;
        }

        // std::cout << " result : " <<  float(((half_float::half*)pim_out)[0]) <<std::endl;
        // std::cout << " result : " <<  float(((half_float::half*)pim_out)[1]) <<std::endl;
        // std::cout << " result : " <<  float(((half_float::half*)pim_out)[2]) <<std::endl;
        // std::cout << " result : " <<  float(((half_float::half*)pim_out)[3]) <<std::endl;
        // std::cout << " result : " <<  float(((half_float::half*)pim_out)[4]) <<std::endl;

        printf("failed ch: ");
        for (int i = ch; i < ch + blocks; i++) {
            if (ch_result[i]) {
                printf("%d ", i);
            }
        }
        printf("\n");
        printf("fail_cnt: %d\n", fail_cnt);
        if (fail_cnt > 0) printf("fail_idx: %d\n", i);
        total_fail_cnt += fail_cnt;
    }
    if (total_fail_cnt)
        printf("FAIL: %d\n", total_fail_cnt);
    else
        printf("PASS\n");

#endif

    free(mode1_h);
    free(mode2_h);
    free(crf_bin_h);
    free(input_host);
    free(weight_host);

    hipFree(mode1_d);
    hipFree(mode2_d);
    hipFree(crf_bin_d);
    hipFree(input_device);
    hipFree(weight_device);
}
