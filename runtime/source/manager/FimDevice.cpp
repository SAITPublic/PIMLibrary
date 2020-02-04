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
FimDevice::FimDevice(FimPrecision precision) : precision_(precision) { LOGI(FIM_DEV, "called"); }

FimDevice::~FimDevice(void) {}

int FimDevice::Initialize(void)
{
    LOGI(FIM_DEV, "called");
    int ret = 0;

    /* open fim device driver */
    ret = OpenDevice();

    return ret;
}

int FimDevice::Deinitialize(void)
{
    LOGI(FIM_DEV, "called");
    int ret = 0;

    /* close fim device driver */
    ret = CloseDevice();

    return ret;
}

int FimDevice::RequestIoctl(void)
{
    LOGI(FIM_DEV, "called");
    int ret = 0;

    /* ioctl by fim command */
    ret = RequestIoctl();

    return ret;
}

int FimDevice::OpenDevice(void)
{
    LOGI(FIM_DEV, "called");
    int ret = 0;

    return ret;
}

int FimDevice::CloseDevice(void)
{
    LOGI(FIM_DEV, "called");
    int ret = 0;

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
