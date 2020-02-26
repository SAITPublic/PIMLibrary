#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"
#include "utility/fim_util.h"

#define LENGTH (1024)

using namespace std;

bool simple_fim_alloc_free()
{
    FimBo fim_weight = {.size = LENGTH * sizeof(_Float16 ), .mem_type = MEM_TYPE_FIM};

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
    FimBo fim_weight = {.size = LENGTH * sizeof(_Float16 ), .mem_type = MEM_TYPE_FIM};

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
    FimBo fim_weight = {.size = LENGTH * sizeof(_Float16 ) * 1024 * 1024, .mem_type = MEM_TYPE_FIM};

    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    while (true) {
        int ret = FimAllocMemory(&fim_weight);
        if (ret) return false;
    }

    FimDeinitialize();

    return true;
}

TEST(UnitTest, simpleFimAllocFree) { EXPECT_TRUE(simple_fim_alloc_free()); }
TEST(UnitTest, FimRepeatAllocateFree) { EXPECT_TRUE(fim_repeat_allocate_free()); }
TEST(UnitTest, FimAllocateExceedBlocksize) { EXPECT_FALSE(fim_allocate_exceed_blocksize()); }
