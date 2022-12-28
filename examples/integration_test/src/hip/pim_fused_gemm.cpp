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

#include "test_fused_gemm.h"

TEST_F(PimFusedGemmTestFixture, hip_pim_fused_gemm_bias_relu_4x1x1024_4x1024x4096_4x4096x1024)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1, 4, 1, 1024, 4096, 1024, true, ACT_RELU, true, NONE) == 0);
}
