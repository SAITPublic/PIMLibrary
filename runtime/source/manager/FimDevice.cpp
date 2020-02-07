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
FimDevice::FimDevice(FimPrecision precision) : precision_(precision) { DLOG(INFO) << "called"; }
FimDevice::~FimDevice(void) {}
int FimDevice::Initialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    /* open fim device driver */
    ret = OpenDevice();

    return ret;
}

int FimDevice::Deinitialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    /* close fim device driver */
    ret = CloseDevice();

    return ret;
}

int FimDevice::RequestIoctl(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    /* ioctl by fim command */
    ret = RequestIoctl();

    return ret;
}

int FimDevice::OpenDevice(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

int FimDevice::CloseDevice(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
