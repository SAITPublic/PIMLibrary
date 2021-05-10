/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_DEVICE_H_
#define _PIM_DEVICE_H_

#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class PimDevice
{
   public:
    PimDevice(PimPrecision precision);
    virtual ~PimDevice(void);

    int initialize(void);
    int deinitialize(void);
    int request_ioctl(void);

   private:
    int open_device(void);
    int close_device(void);

    static constexpr char pim_drv_name_[] = "/dev/pim_drv";
    size_t grf_size_;
    size_t grf_cnt_;
    size_t channel_cnt_;
    PimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_DEVICE_H_ */
