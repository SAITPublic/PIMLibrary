#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "half.hpp"
#include "utility/fim_dump.hpp"
#include "utility/fim_profile.h"

#define LENGTH (128 * 1024)

#ifdef DEBUG_FIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using namespace std;
using half_float::half;

int fim_elt_mul_1(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input0 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* fim_input1 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/elt_mul/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_mul/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_mul/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_mul/output_256KB.dat";

    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (ELT_MUL) */
        FimExecuteMul(device_output, fim_input0, fim_input1, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, FIM_TO_HOST);

        //    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);
        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data,
                                     host_output->size / sizeof(half));
    }
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input0);
    FimDestroyBo(host_input1);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input0);
    FimDestroyBo(fim_input1);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_elt_mul_2(bool block)
{
    int ret = 0;

    FimBo host_input0 = {.mem_type = MEM_TYPE_HOST, .size = LENGTH * sizeof(half)};
    FimBo host_input1 = {.mem_type = MEM_TYPE_HOST, .size = LENGTH * sizeof(half)};
    FimBo host_output = {.mem_type = MEM_TYPE_HOST, .size = LENGTH * sizeof(half)};
    FimBo golden_output = {.mem_type = MEM_TYPE_HOST, .size = LENGTH * sizeof(half)};
    FimBo fim_input0 = {.mem_type = MEM_TYPE_FIM, .size = LENGTH * sizeof(half)};
    FimBo fim_input1 = {.mem_type = MEM_TYPE_FIM, .size = LENGTH * sizeof(half)};
    FimBo device_output = {.mem_type = MEM_TYPE_FIM, .size = LENGTH * sizeof(half)};

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate memory */
    FimAllocMemory(&host_input0);
    FimAllocMemory(&host_input1);
    FimAllocMemory(&host_output);
    FimAllocMemory(&golden_output);
    FimAllocMemory(&fim_input0);
    FimAllocMemory(&fim_input1);
    FimAllocMemory(&device_output);

    std::string test_vector_data = TEST_VECTORS_DATA;
    std::string input0 = test_vector_data + "load/elt_mul/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_mul/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_mul/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_mul/output_256KB.dat";

    /* Initialize the input, weight, output data */
    load_data(input0.c_str(), (char*)host_input0.data, host_input0.size);
    load_data(input1.c_str(), (char*)host_input1.data, host_input1.size);
    load_data(output.c_str(), (char*)golden_output.data, golden_output.size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(&fim_input0, &host_input0, HOST_TO_FIM);
    FimCopyMemory(&fim_input1, &host_input1, HOST_TO_FIM);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (ELT_MUL) */
        FimExecuteMul(&device_output, &fim_input0, &fim_input1, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(&host_output, &device_output, FIM_TO_HOST);

        //    ret = compare_data((char*)golden_output.data, (char*)host_output.data, host_output.size);
        ret =
            compare_data_round_off((half*)golden_output.data, (half*)host_output.data, host_output.size / sizeof(half));
    }
    //    dump_data(output_dump.c_str(), (char*)host_output.data, host_output.size);

    /* __FIM_API__ call : Free memory */
    FimFreeMemory(&host_input0);
    FimFreeMemory(&host_input1);
    FimFreeMemory(&host_output);
    FimFreeMemory(&golden_output);
    FimFreeMemory(&device_output);
    FimFreeMemory(&fim_input0);
    FimFreeMemory(&fim_input1);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_elt_mul_3(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input0 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* fim_input1 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/elt_mul/input0_512KB.dat";
    std::string input1 = test_vector_data + "load/elt_mul/input1_512KB.dat";
    std::string output = test_vector_data + "load/elt_mul/output_512KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_mul/output_512KB.dat";

    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (ELT_MUL) */
        FimExecuteMul(device_output, fim_input0, fim_input1, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, FIM_TO_HOST);

        //    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);
        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data,
                                     host_output->size / sizeof(half));
    }
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input0);
    FimDestroyBo(host_input1);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input0);
    FimDestroyBo(fim_input1);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_elt_mul_4(bool block)
{
    int ret = 0;
    int in_size = 128 * 768;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimDesc* fim_desc = FimCreateDesc(1, 1, 1, in_size, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* fim_input0 = FimCreateBo(fim_desc, MEM_TYPE_FIM);
    FimBo* fim_input1 = FimCreateBo(fim_desc, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;
    /* Initialize the input, weight, output data */
    std::string input0 = test_vector_data + "load/elt_mul/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_mul/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_mul/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_mul/output_256KB.dat";

    /* Initialize the input, weight, output data */
    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (ELT_MUL) */
        FimExecuteMul(device_output, fim_input0, fim_input1, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, FIM_TO_HOST);

        //    ret = compare_data((char*)golden_output->data, (char*)host_output->data, in_size * sizeof(half));
        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, in_size);
    }
    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input0);
    FimDestroyBo(host_input1);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input0);
    FimDestroyBo(fim_input1);
    FimDestroyDesc(fim_desc);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_elt_mul_profile(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_input0 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* fim_input1 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/elt_mul/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_mul/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_mul/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_mul/output_256KB.dat";

    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);

    FimExecuteDummy();
/* __FIM_API__ call : Execute FIM kernel (ELT_MUL) */
#ifdef TARGET
    int iter;
    FIM_PROFILE_TICK(ELT_MUL_1);
    for (iter = 0; iter < 1000; iter++) {
#endif
        FimExecuteMul(device_output, fim_input0, fim_input1, nullptr, block);
        if (!block) FimSynchronize();

#ifdef TARGET
    }
    FIM_PROFILE_TOCK(ELT_MUL_1);
    // printf("[ %d execution time ]\n", iter);
#endif
    FimCopyMemory(host_output, device_output, FIM_TO_HOST);

    //    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);
    ret =
        compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / sizeof(half));

    //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input0);
    FimDestroyBo(host_input1);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input0);
    FimDestroyBo(fim_input1);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, FimEltMul1Sync) { EXPECT_TRUE(fim_elt_mul_1(true) == 0); }
TEST(HIPIntegrationTest, FimEltMul1Async) { EXPECT_TRUE(fim_elt_mul_1(false) == 0); }
TEST(HIPIntegrationTest, FimEltMul2Sync) { EXPECT_TRUE(fim_elt_mul_2(true) == 0); }
TEST(HIPIntegrationTest, FimEltMul2Async) { EXPECT_TRUE(fim_elt_mul_2(false) == 0); }
TEST(HIPIntegrationTest, FimEltMul3Sync) { EXPECT_TRUE(fim_elt_mul_3(true) == 0); }
TEST(HIPIntegrationTest, FimEltMul3Async) { EXPECT_TRUE(fim_elt_mul_3(false) == 0); }
TEST(HIPIntegrationTest, FimEltMul4Sync) { EXPECT_TRUE(fim_elt_mul_4(true) == 0); }
TEST(HIPIntegrationTest, FimEltMul4Async) { EXPECT_TRUE(fim_elt_mul_4(false) == 0); }
TEST(HIPIntegrationTest, FimEltMulProfileSync) { EXPECT_TRUE(fim_elt_mul_profile(true) == 0); }
