/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_DEVICE_H_
#define _FIM_DEVICE_H_

#include "fim_data_types.h"

namespace fim
{
namespace runtime
{
namespace manager
{
class FimDevice
{
   public:
    FimDevice(FimPrecision precision);
    virtual ~FimDevice(void);

    int initialize(void);
    int deinitialize(void);
    int request_ioctl(void);

   private:
    int open_device(void);
    int close_device(void);

    static constexpr char fim_drv_name_[] = "/dev/fim_drv";
    size_t grf_size_;
    size_t grf_cnt_;
    size_t channel_cnt_;
    FimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_DEVICE_H_ */
