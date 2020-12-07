#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "half.hpp"
#include "hip/hip_runtime.h"
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

int fim_elt_add_1(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);

    FimBo* fim_input1 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* fim_input0 = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/elt_add/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_add/output_256KB.dat";

    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
        FimExecuteAdd(device_output, fim_input0, fim_input1, nullptr, block);

        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, FIM_TO_HOST);

        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data,
                                     host_output->size / sizeof(half), 0.05);
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

int fim_elt_add_2(bool block)
{
    int ret = 0;

    FimBo host_input0 = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo host_input1 = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo host_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo golden_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo fim_input1 = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_FIM};
    FimBo fim_input0 = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_FIM};
    FimBo device_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_FIM};

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
    std::string input0 = test_vector_data + "load/elt_add/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_add/output_256KB.dat";

    /* Initialize the input, weight, output data */
    load_data(input0.c_str(), (char*)host_input0.data, host_input0.size);
    load_data(input1.c_str(), (char*)host_input1.data, host_input1.size);
    load_data(output.c_str(), (char*)golden_output.data, golden_output.size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(&fim_input0, &host_input0, HOST_TO_FIM);
    FimCopyMemory(&fim_input1, &host_input1, HOST_TO_FIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
        FimExecuteAdd(&device_output, &fim_input0, &fim_input1, nullptr, block);

        if (!block) FimSynchronize();

        FimCopyMemory(&host_output, &device_output, FIM_TO_HOST);

        ret = compare_data_round_off((half*)golden_output.data, (half*)host_output.data,
                                     host_output.size / sizeof(half), 0.05);
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

int fim_elt_add_3(bool block)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_output = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* fim_input0 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* fim_input1 = FimCreateBo(LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/elt_add/input0_512KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_512KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_512KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_add/output_512KB.dat";

    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);
    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
        FimExecuteAdd(device_output, fim_input0, fim_input1, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, FIM_TO_HOST);

        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data,
                                     host_output->size / sizeof(half), 0.05);
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

int fim_elt_add_4(bool block)
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
    std::string input0 = test_vector_data + "load/elt_add/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/elt_add/output_256KB.dat";

    /* Initialize the input, weight, output data */
    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
        FimExecuteAdd(device_output, fim_input0, fim_input1, nullptr, block);
        if (!block) FimSynchronize();

        FimCopyMemory(host_output, device_output, FIM_TO_HOST);
        ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, in_size, 0.05);
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

int fim_elt_add_profile(bool block, int len)
{
    int ret = 0;
    int length = len;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input0 = FimCreateBo(length, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_input1 = FimCreateBo(length, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(length, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(length, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);

    FimBo* fim_input1 = FimCreateBo(length, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(length, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* fim_input0 = FimCreateBo(length, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input0 = test_vector_data + "load/elt_add/input0_32768KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_32768KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_32768KB.dat";
    //    std::string output_dump = test_vector_data + "dump/elt_add/output_32768KB.dat";

    load_data(input0.c_str(), (char*)host_input0->data, host_input0->size);
    load_data(input1.c_str(), (char*)host_input1->data, host_input1->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);

    //    FimExecuteDummy();
    FimExecuteAdd(device_output, fim_input0, fim_input1, nullptr, block);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 0.0f;
    hipDeviceSynchronize();

    hipEventRecord(start, nullptr);
/* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
#ifdef TARGET
    int iter;
    //    FIM_PROFILE_TICK(ELT_ADD_1);
    for (iter = 0; iter < 100; iter++) {
#endif
        FimExecuteAdd(device_output, fim_input0, fim_input1, nullptr, block);
#ifdef TARGET
    }
//    if (!block) FimSynchronize();
//    FIM_PROFILE_TOCK(ELT_ADD_1);
//    printf("[ %d execution time ]\n", iter);

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs / 100);
#endif
    FimCopyMemory(host_output, device_output, FIM_TO_HOST);

    ret = compare_data_round_off((half*)golden_output->data, (half*)host_output->data, host_output->size / sizeof(half),
                                 0.05);

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

TEST(HIPIntegrationTest, FimEltAdd1Sync) { EXPECT_TRUE(fim_elt_add_1(true) == 0); }
TEST(HIPIntegrationTest, FimEltAdd1Async) { EXPECT_TRUE(fim_elt_add_1(false) == 0); }
TEST(HIPIntegrationTest, FimEltAdd2Sync) { EXPECT_TRUE(fim_elt_add_2(true) == 0); }
TEST(HIPIntegrationTest, FimEltAdd2ASync) { EXPECT_TRUE(fim_elt_add_2(false) == 0); }
TEST(HIPIntegrationTest, FimEltAdd3Sync) { EXPECT_TRUE(fim_elt_add_3(true) == 0); }
TEST(HIPIntegrationTest, FimEltAdd3Async) { EXPECT_TRUE(fim_elt_add_3(false) == 0); }
TEST(HIPIntegrationTest, FimEltAdd4Sync) { EXPECT_TRUE(fim_elt_add_4(true) == 0); }
TEST(HIPIntegrationTest, FimEltAdd4ASync) { EXPECT_TRUE(fim_elt_add_4(false) == 0); }
TEST(HIPIntegrationTest, FimEltAddProfile1Sync) { EXPECT_TRUE(fim_elt_add_profile(true, (128 * 1024)) == 0); }
TEST(HIPIntegrationTest, FimEltAddProfile1Async) { EXPECT_TRUE(fim_elt_add_profile(false, (128 * 1024)) == 0); }
// TEST(HIPIntegrationTest, FimEltAddProfile2Async) { EXPECT_TRUE(fim_elt_add_profile(false, (256 * 1024)) == 0); }
// TEST(HIPIntegrationTest, FimEltAddProfile3Async) { EXPECT_TRUE(fim_elt_add_profile(false, (512 * 1024)) == 0); }
// TEST(HIPIntegrationTest, FimEltAddProfile4Async) { EXPECT_TRUE(fim_elt_add_profile(false, (1024 * 1024)) == 0); }
// TEST(HIPIntegrationTest, FimEltAddProfile5Async) { EXPECT_TRUE(fim_elt_add_profile(false, (2048 * 1024)) == 0); }
// TEST(HIPIntegrationTest, FimEltAddProfile6Async) { EXPECT_TRUE(fim_elt_add_profile(false, (4096 * 1024)) == 0); }
// TEST(HIPIntegrationTest, FimEltAddProfile7Async) { EXPECT_TRUE(fim_elt_add_profile(false, (8192 * 1024)) == 0); }
// TEST(HIPIntegrationTest, FimEltAddProfile8Async) { EXPECT_TRUE(fim_elt_add_profile(false, (16384 * 1024)) == 0); }