#include "manager/FimDevice.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>

namespace fim
{
namespace runtime
{
namespace manager
{
FimDevice::FimDevice(FimPrecision precision) : precision_(precision)
{
    std::cout << "fim::runtime::manager FimDevice creator call" << std::endl;
}

FimDevice::~FimDevice(void) {}

int FimDevice::Initialize(void)
{
    std::cout << "fim::runtime::manager FimDiver::Initialize call" << std::endl;

    int ret = 0;

    /* open fim device driver */
    ret = OpenDevice();

    return ret;
}

int FimDevice::Deinitialize(void)
{
    std::cout << "fim::runtime::manager FimDevice::Deinitialize call" << std::endl;
    int ret = 0;

    /* close fim device driver */
    ret = CloseDevice();

    return ret;
}

int FimDevice::RequestIoctl(void)
{
    std::cout << "fim::runtime::manager FimDevice::RequestIoctl call" << std::endl;

    int ret = 0;

    /* ioctl by fim command */
    ret = RequestIoctl();

    return ret;
}

int FimDevice::OpenDevice(void)
{
    std::cout << "fim::runtime::manager FimDevice::OpenDevice call" << std::endl;

    int ret = 0;

    return ret;
}

int FimDevice::CloseDevice(void)
{
    std::cout << "fim::runtime::manager FimDevice::CloseDevice call" << std::endl;

    int ret = 0;

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
