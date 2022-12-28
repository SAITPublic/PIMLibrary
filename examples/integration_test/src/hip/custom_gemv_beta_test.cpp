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

#include "test_custom_gemv_beta.h"

TEST_F(PimGemvTestFixture, hip_custom_gemv_beta_1x320_i_x_w)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 320, 1, 1280, I_X_W, true) == 0);
}

TEST_F(PimGemvTestFixture, hip_custom_gemv_beta_320x1_w_x_i)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(320, 1, 1280, 1, W_X_I, true) == 0);
}
