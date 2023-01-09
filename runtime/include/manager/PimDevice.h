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

#ifndef _PIM_DEVICE_H_
#define _PIM_DEVICE_H_

#include "manager/PimInfo.h"
#include "utility/pim_util.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class PimDevice
{
   public:
    PimDevice(void) { ::get_pim_block_info(&pbi_); }
    virtual ~PimDevice(void) {}
    PimBlockInfo* get_pim_block_info(void) { return &pbi_; }
#if 0 /* TODO: enable PIM Device Driver */
    int init_device(void) = 0;
    int deinit_device(void) = 0;
    int set_property(void) = 0;
    int get_property(void) = 0;
    int open_device(void) = 0;
    int close_device(void) = 0;
#endif
   private:
    PimBlockInfo pbi_;
};
} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_DEVICE_H_ */
