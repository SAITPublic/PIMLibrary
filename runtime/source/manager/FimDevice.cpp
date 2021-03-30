/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/FimDevice.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimDevice::FimDevice(FimPrecision precision) : precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
}
FimDevice::~FimDevice(void) {}
int FimDevice::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    /* open fim device driver */
    ret = open_device();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimDevice::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    /* close fim device driver */
    ret = close_device();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimDevice::request_ioctl(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    /* ioctl by fim command */
    ret = request_ioctl();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimDevice::open_device(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimDevice::close_device(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
