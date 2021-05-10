/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_CONTROL_MANAGER_H_
#define _PIM_CONTROL_MANAGER_H_

#include "pim_data_types.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class PimDevice;

class PimControlManager
{
   public:
    PimControlManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision);
    virtual ~PimControlManager(void);

    int initialize(void);
    int deinitialize(void);
    int program_crf_code(void);
    int set_pim_mode(void);

   private:
    PimDevice* pim_device_;
    PimRuntimeType rt_type_;
    PimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_CONTROL_MANAGER_H_ */
