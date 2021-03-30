/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/FimControlManager.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "fim_data_types.h"
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimControlManager::FimControlManager(FimDevice* fim_device, FimRuntimeType rt_type, FimPrecision precision)
    : fim_device_(fim_device), rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

FimControlManager::~FimControlManager(void) { DLOG(INFO) << "[START] " << __FUNCTION__ << " called"; }
int FimControlManager::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimControlManager::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimControlManager::program_crf_code(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    int ret = 0;

    /* ioctl by fim command */
    ret = fim_device_->request_ioctl();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimControlManager::set_fim_mode(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    int ret = 0;

    /* ioctl by fim command */
    ret = fim_device_->request_ioctl();
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
