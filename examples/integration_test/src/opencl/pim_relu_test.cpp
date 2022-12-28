
/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "test_relu.h"

TEST_F(PimReluTestFixture, ocl_pim_relu_256x1024)
{
    SetUp(RT_TYPE_OPENCL);
    EXPECT_TRUE(ExecuteTest(256 * 1024) == 0);
}
