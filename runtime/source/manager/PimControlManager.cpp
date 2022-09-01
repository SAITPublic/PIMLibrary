/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/PimControlManager.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "pim_data_types.h"
#include "utility/pim_log.h"

namespace pim
{
namespace runtime
{
namespace manager
{
PimControlManager::PimControlManager(PimDevice* pim_device, PimRuntimeType rt_type, PimPrecision precision)
    : pim_device_(pim_device), rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

PimControlManager::~PimControlManager(void) { DLOG(INFO) << "[START] " << __FUNCTION__ << " called"; }
int PimControlManager::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimControlManager::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    int ret = 0;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimControlManager::program_crf_code(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    int ret = 0;

    /* ioctl by pim command */
    // ret = pim_device_->request_ioctl();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int PimControlManager::set_pim_mode(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    int ret = 0;

    /* ioctl by pim command */
    // ret = pim_device_->request_ioctl();
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */
