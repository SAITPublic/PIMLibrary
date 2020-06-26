#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"
#include "utility/fim_dump.hpp"
#include "utility/fim_util.h"

#define LENGTH (1024)

using namespace std;

bool fim_memcpy_test(void)
{
    size_t size = 5 * 2;
    char* data_a;
    char* data_b;
    int ret = 0;

    FimInitialize();

    FimAllocMemory((void**)&data_a, size, MEM_TYPE_HOST);
    FimAllocMemory((void**)&data_b, size, MEM_TYPE_HOST);

    for (int i = 0; i < size; i++) {
        data_a[i] = 'C';
        data_b[i] = 'D';
    }

    FimCopyMemory((void*)data_b, (void*)data_a, size, HOST_TO_HOST);

    ret = compare_data(data_a, data_b, size);
    if (ret != 0) {
        std::cout << "data is different" << std::endl;
        return false;
    }

    FimFreeMemory(data_a, MEM_TYPE_HOST);
    FimFreeMemory(data_b, MEM_TYPE_HOST);

    FimDeinitialize();

    return true;
}

bool simple_fim_alloc_free()
{
    FimBo fim_weight = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_FIM};

    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    int ret = FimAllocMemory(&fim_weight);
    if (ret) return false;

    ret = FimFreeMemory(&fim_weight);
    if (ret) return false;

    FimDeinitialize();

    return true;
}

bool fim_repeat_allocate_free(void)
{
    FimBo fim_weight = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_FIM};

    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    int i = 0;
    while (i < 100) {
        int ret = FimAllocMemory(&fim_weight);
        if (ret) return false;
        ret = FimFreeMemory(&fim_weight);
        if (ret) return false;
        i++;
    }

    FimDeinitialize();

    return true;
}

bool fim_allocate_exceed_blocksize(void)
{
    FimBo fim_weight = {.size = LENGTH * sizeof(half) * 1024 * 1024, .mem_type = MEM_TYPE_FIM};

    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    while (true) {
        int ret = FimAllocMemory(&fim_weight);
        if (ret) return false;
    }

    FimDeinitialize();

    return true;
}

TEST(UnitTest, FimMemCopyTest) { EXPECT_TRUE(fim_memcpy_test()); }
TEST(UnitTest, simpleFimAllocFree) { EXPECT_TRUE(simple_fim_alloc_free()); }
TEST(UnitTest, FimRepeatAllocateFree) { EXPECT_TRUE(fim_repeat_allocate_free()); }
TEST(UnitTest, FimAllocateExceedBlocksize) { EXPECT_FALSE(fim_allocate_exceed_blocksize()); }
