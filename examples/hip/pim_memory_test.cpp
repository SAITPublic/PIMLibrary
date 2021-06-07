/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "hip/hip_fp16.h"
#include "pim_runtime_api.h"
#include "utility/pim_dump.hpp"
#include "utility/pim_util.h"

#define LENGTH (1024)

using namespace std;

bool pim_memcpy_test(void)
{
    size_t size = 5 * 2;
    char* data_a;
    char* data_b;
    int ret = 0;

    PimInitialize();

    PimAllocMemory((void**)&data_a, size, MEM_TYPE_HOST);
    PimAllocMemory((void**)&data_b, size, MEM_TYPE_HOST);

    for (int i = 0; i < size; i++) {
        data_a[i] = 'C';
        data_b[i] = 'D';
    }

    PimCopyMemory((void*)data_b, (void*)data_a, size, HOST_TO_HOST);

    ret = compare_data(data_a, data_b, size);
    if (ret != 0) {
        std::cout << "data is different" << std::endl;
        return false;
    }

    PimFreeMemory(data_a, MEM_TYPE_HOST);
    PimFreeMemory(data_b, MEM_TYPE_HOST);

    PimDeinitialize();

    return true;
}

bool simple_pim_alloc_free()
{
    PimBo pim_weight = {.mem_type = MEM_TYPE_PIM, .size = LENGTH * sizeof(half)};

    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    int ret = PimAllocMemory(&pim_weight);
    if (ret) return false;

    ret = PimFreeMemory(&pim_weight);
    if (ret) return false;

    PimDeinitialize();

    return true;
}

bool pim_repeat_allocate_free(void)
{
    PimBo pim_weight = {.mem_type = MEM_TYPE_PIM, .size = LENGTH * sizeof(half)};

    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    int i = 0;
    while (i < 100) {
        int ret = PimAllocMemory(&pim_weight);
        if (ret) return false;
        ret = PimFreeMemory(&pim_weight);
        if (ret) return false;
        i++;
    }

    PimDeinitialize();

    return true;
}

bool pim_allocate_exceed_blocksize(void)
{
    std::vector<PimBo> pimObjPtr;

    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    int ret;
    while (true) {
        PimBo pim_weight = {.mem_type = MEM_TYPE_PIM, .size = LENGTH * sizeof(half) * 1024 * 1024};
        ret = PimAllocMemory(&pim_weight);
        if (ret) break;
        pimObjPtr.push_back(pim_weight);
    }

    for (int i = 0; i < pimObjPtr.size(); i++) PimFreeMemory(&pimObjPtr[i]);

    PimDeinitialize();

    if (ret) return false;

    return true;
}

TEST(UnitTest, PimMemCopyTest) { EXPECT_TRUE(pim_memcpy_test()); }
TEST(UnitTest, simplePimAllocFree) { EXPECT_TRUE(simple_pim_alloc_free()); }
TEST(UnitTest, PimRepeatAllocateFree) { EXPECT_TRUE(pim_repeat_allocate_free()); }
TEST(UnitTest, PimAllocateExceedBlocksize) { EXPECT_FALSE(pim_allocate_exceed_blocksize()); }
