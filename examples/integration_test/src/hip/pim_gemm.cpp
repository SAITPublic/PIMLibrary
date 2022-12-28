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

#include "test_gemm.h"

TEST_F(PimGemmTestFixture, hip_pim_gemm_4096x1024_1024x1_w_x_i)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 1, 1024, 1, 4096, 1, W_X_I, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x4096x1024_2x1024x1_w_x_i)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 2, 1024, 1, 4096, 1, W_X_I, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_1x1024_1024x4096_i_x_w)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 1, 1, 1024, 1, 4096, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1024_1024x4096_i_x_w)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 1, 2, 1024, 2, 4096, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1x1024_2x1024x4096_i_x_w)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 2, 1, 1024, 1, 4096, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x2x1024_2x1024x4096_i_x_w)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 2, 2, 1024, 2, 4096, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x256_64x256x64_i_x_w)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 64, 1, 256, 1, 64, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x1024_64x1024x64_i_x_w)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 64, 1, 1024, 1, 64, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_4x1x4096_4x4096x1024_i_x_w)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 4, 1, 4096, 1, 1024, I_X_W, false, true, NONE) == 0);
}
#if 0
/* TODO:check */

TEST_F(PimGemmTestFixture, hip_pim_gemm_w_x_i_4096x240_240x1)
{
    EXPECT_TRUE(ExecuteTest(1, 1, 240, 1, 4096, 1, W_X_I) == 0);
}

TEST_F(PimGemmTestFixture, hip_pim_gemm_w_x_i_4096x1024_1024x2)
{
    EXPECT_TRUE(ExecuteTest(1, 1, 1024, 2, 4096, 2, W_X_I) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_8x1x4096_8x4096x1024)
{
    EXPECT_TRUE(ExecuteTest(1, 8, 1, 4096, 1, 1024) == 0);
}
#endif
TEST_F(PimGemmTestFixture, hip_pim_gemm_1x1024_1024x4096_reordering_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 1, 1, 1024, 1, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1024_1024x4096_reordering_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 1, 2, 1024, 2, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1x1024_2x1024x4096_reordering_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 2, 1, 1024, 1, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x2x1024_2x1024x4096_reordering_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 2, 2, 1024, 2, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x256_64x256x64_reordering_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 64, 1, 256, 1, 64, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x1024_64x1024x64_reordering_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 64, 1, 1024, 1, 64, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_4x1x4096_4x4096x1024_reordering_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 4, 1, 4096, 1, 1024, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_1x1024_1024x4096_reordering_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 1, 1, 1024, 1, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1024_1024x4096_reordering_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 1, 2, 1024, 2, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1x1024_2x1024x4096_reordering_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 2, 1, 1024, 1, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x2x1024_2x1024x4096_reordering_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 2, 2, 1024, 2, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x256_64x256x64_reordering_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 64, 1, 256, 1, 64, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x1024_64x1024x64_reordering_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 64, 1, 1024, 1, 64, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_4x1x4096_4x4096x1024_reordering_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReordering(1, 4, 1, 4096, 1, 1024, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_1x1024_1024x4096_reordering_on_device_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 1, 1, 1024, 1, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1024_1024x4096_reordering_on_device_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 1, 2, 1024, 2, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1x1024_2x1024x4096_reordering_on_device_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 2, 1, 1024, 1, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x2x1024_2x1024x4096_reordering_on_device_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 2, 2, 1024, 2, 4096, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x256_64x256x64_reordering_on_device_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 64, 1, 256, 1, 64, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x1024_64x1024x64_reordering_on_device_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 64, 1, 1024, 1, 64, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_4x1x4096_4x4096x1024_reordering_on_device_from_device)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 4, 1, 4096, 1, 1024, true) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_1x1024_1024x4096_reordering_on_device_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 1, 1, 1024, 1, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1024_1024x4096_reordering_on_device_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 1, 2, 1024, 2, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x1x1024_2x1024x4096_reordering_on_device_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 2, 1, 1024, 1, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_2x2x1024_2x1024x4096_reordering_on_device_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 2, 2, 1024, 2, 4096, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x256_64x256x64_reordering_on_device_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 64, 1, 256, 1, 64, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_64x1x1024_64x1024x64_reordering_on_device_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 64, 1, 1024, 1, 64, false) == 0);
}
TEST_F(PimGemmTestFixture, hip_pim_gemm_4x1x4096_4x4096x1024_reordering_on_device_from_host)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTestExplicitReorderingOnDevice(1, 4, 1, 4096, 1, 1024, false) == 0);
}
