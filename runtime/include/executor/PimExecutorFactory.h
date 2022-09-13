/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_EXECUTOR_FACTORY_H_
#define _PIM_EXECUTOR_FACTORY_H_

#include "executor/IPimExecutor.h"
#include "executor/hip/HipPimExecutor.h"
#include "executor/ocl/OclPimExecutor.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace executor
{
class PimExecutorFactory
{
   public:
    static std::shared_ptr<IPimExecutor> getPimExecutor(pim::runtime::manager::PimManager* pim_manager,
                                                        PimRuntimeType rt_type, PimPrecision precision)
    {
        if (rt_type == RT_TYPE_HIP) {
            return std::make_shared<HipPimExecutor>(pim_manager, precision);
        } else if (rt_type == RT_TYPE_OPENCL) {
            return std::make_shared<OclPimExecutor>(pim_manager, precision);
        } else {
            throw std::invalid_argument("invalid type of runtime");
        }
    }
};
}  // namespace executor
}  // namespace runtime
}  // namespace pim

#endif  // _PIM_EXECUTOR_FACTORY_H_
