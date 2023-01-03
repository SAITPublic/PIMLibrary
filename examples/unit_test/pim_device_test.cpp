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

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <random>
#include "hip/hip_fp16.h"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

using namespace std;

bool test_set_get_device()
{
    int ret = 0;
    uint32_t set_device_id = 1;
    uint32_t get_device_id = 0;

    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    ret = PimSetDevice(set_device_id);
    if (ret != 0) {
        std::cout << "fail to set PimDevice" << std::endl;
        return false;
    }

    ret = PimGetDevice(&get_device_id);
    if (ret != 0) {
        std::cout << "fail to get PimDevice" << std::endl;
        return false;
    }

    if (set_device_id != get_device_id) {
        std::cout << "invalid PimDevice id" << std::endl;
        return false;
    }

    PimDeinitialize();

    return true;
}

TEST(UnitTest, hip_PimSetGetDeviceTest) { EXPECT_TRUE(test_set_get_device()); }
