
/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "test_eltwise.h"

// Add Test Cases
TEST_F(PimEltwiseTestFixture, ocl_pim_eltWise_add_256x1024x1x1x1)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(256 * 1024, 1, 1, 1, "add") == 0);
}
// Mul Test Cases
TEST_F(PimEltwiseTestFixture, ocl_pim_eltWise_mul_256x1024x1x1x1)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(256 * 1024, 1, 1, 1, "mul") == 0);
}
