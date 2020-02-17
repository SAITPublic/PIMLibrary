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
    DLOG(INFO) << "called";
}

FimControlManager::~FimControlManager(void) { DLOG(INFO) << "called"; }
int FimControlManager::initialize(void)
{
    DLOG(INFO) << "called";

    int ret = 0;

    return ret;
}

int FimControlManager::deinitialize(void)
{
    DLOG(INFO) << "called";

    int ret = 0;

    return ret;
}

int FimControlManager::program_crf_code(void)
{
    DLOG(INFO) << "called";

    int ret = 0;

    /* ioctl by fim command */
    ret = fim_device_->request_ioctl();

    return ret;
}

int FimControlManager::set_fim_mode(void)
{
    DLOG(INFO) << "called";

    int ret = 0;

    /* ioctl by fim command */
    ret = fim_device_->request_ioctl();

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
