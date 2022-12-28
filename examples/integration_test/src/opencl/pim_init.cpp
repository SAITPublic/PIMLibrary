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
#include <random>
#include "utility/pim_profile.h"
#include "utility/pim_util.h"
#include "pim_runtime_api.h"

using namespace std;

bool create_ocl_kernel_binary(void)
{
    /* generate PIM ocl kernel binary during initialization */
    PimInitialize(RT_TYPE_OPENCL, PIM_FP16);
    PimDeinitialize();

    return true;
}

TEST(UnitTest, create_ocl_kernel_binary_test) { EXPECT_TRUE(create_ocl_kernel_binary()); }
