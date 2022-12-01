/**
 * @file PimCompilerDriver.cpp
 * @brief Source file for PimCompilerDriver base implementation
 *
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system, or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */
#include "executor/PimCompilerDriver.h"
#include "executor/hip/PimCompilerHipDriver.hpp"

namespace pim
{
namespace runtime
{
namespace pimc_driver
{
#if PIM_COMPILER_ENABLE == 1
std::shared_ptr<Driver> Driver::create_driver(PimTarget* target)
{
    if (target->runtime == PimRuntimeType::RT_TYPE_HIP) {
        return std::make_shared<HIPDriver>();
    } else {
        // else  TODO: OCL
    }
}
#endif
}  // namespace pimc_driver
}  // namespace runtime
}  // namespace pim
