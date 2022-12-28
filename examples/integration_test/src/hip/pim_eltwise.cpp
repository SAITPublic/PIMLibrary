/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "test_eltwise.h"

// Add test cases
TEST_F(PimEltwiseTestFixture, hip_pim_eltWise_add_1x1024x1x1x1)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1 * 1024, 1, 1, 1, "add") == 0);
}
TEST_F(PimEltwiseTestFixture, hip_pim_eltWise_add_128x1024x1x1x1)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(128 * 1024, 1, 1, 1, "add") == 0);
}
TEST_F(PimEltwiseTestFixture, hip_pim_eltWise_add_256x1024x1x1x1)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(256 * 1024, 1, 1, 1, "add") == 0);
}
TEST_F(PimEltwiseTestFixture, hip_pim_eltWise_add_1x1x1x128x768)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 1, 1, 128 * 768, "add") == 0);
}

// Mul Test Cases
TEST_F(PimEltwiseTestFixture, hip_pim_eltWise_mul_256x1024x1x1x1)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(256 * 1024, 1, 1, 1, "mul") == 0);
}
TEST_F(PimEltwiseTestFixture, hip_pim_eltWise_mul_1x1x1x257x1024)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 1, 1, 257 * 1024, "mul") == 0);
}
TEST_F(PimEltwiseTestFixture, hip_pim_eltWise_mul_1x1x1x256x1024)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 1, 256, 1024, "mul") == 0);
}
