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
// OpenCL Test cases
TEST_F(PimGemmTestFixture, ocl_pim_gemm_1x1024_1024x4096)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1, 1, 1, 1024, 1, 4096, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, ocl_pim_gemm_8x1024_1024x4096)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1, 1, 8, 1024, 8, 4096, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, ocl_pim_gemm_4x1x1024_4x1024x4096)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1, 4, 1, 1024, 1, 4096, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, ocl_pim_gemm_4x8x1024_4x1024x4096)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1, 4, 8, 1024, 8, 4096, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, ocl_pim_gemm_64x1x256_64x256x64)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1, 64, 1, 256, 1, 64, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, ocl_pim_gemm_64x1x1024_64x1024x64)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1, 64, 1, 1024, 1, 64, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, ocl_pim_gemm_4x1x4096_4x4096x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1, 4, 1, 4096, 1, 1024, I_X_W, false, true, NONE) == 0);
}
TEST_F(PimGemmTestFixture, ocl_pim_gemm_8x1x4096_8x4096x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(1, 8, 1, 4096, 1, 1024, I_X_W, false, true, NONE) == 0);
}
