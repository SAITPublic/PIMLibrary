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

#ifndef _PIM_MEMORY_MANAGER_FACTORY_H_
#define _PIM_MEMORY_MANAGER_FACTORY_H_

#include "manager/IPimMemoryManager.h"
#include "manager/PimDevice.h"
#include "manager/PimInfo.h"
#include "manager/hip/HipMemoryManager.h"
#include "manager/ocl/OclMemoryManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class PimMemoryManagerFactory
{
   public:
    static std::shared_ptr<IPimMemoryManager> getPimMemoryManager(std::shared_ptr<PimDevice> pim_device,
                                                                  PimRuntimeType rt_type, PimPrecision precision)
    {
        if (rt_type == RT_TYPE_HIP) {
            return std::make_shared<HipMemoryManager>(pim_device, precision);
        } else if (rt_type == RT_TYPE_OPENCL) {
            return std::make_shared<OclMemoryManager>(pim_device, precision);
        } else {
            throw std::invalid_argument("invalid type of runtime");
        }
    }
};
}  // namespace manager
}  // namespace runtime
}  // namespace pim

#endif  // _PIM_MEMORY_MANAGER_FACTORY_H_
