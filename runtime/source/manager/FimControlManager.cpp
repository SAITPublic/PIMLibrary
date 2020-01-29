#include "manager/FimControlManager.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "fim_data_types.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimControlManager::FimControlManager(FimDevice* fimDevice, FimRuntimeType rtType, FimPrecision precision)
    : fimDevice_(fimDevice), rtType_(rtType), precision_(precision)
{
    std::cout << "fim::runtime::manager FimControlManager creator call" << std::endl;
}

FimControlManager::~FimControlManager(void)
{
    std::cout << "fim::runtime::manager FimControlManager destroyer call" << std::endl;
}

int FimControlManager::Initialize(void)
{
    std::cout << "fim::runtime::manager FimControlManager::Initialize call" << std::endl;

    int ret = 0;

    return ret;
}

int FimControlManager::Deinitialize(void)
{
    std::cout << "fim::runtime::manager FimControlManager::Deinitialize call" << std::endl;

    int ret = 0;

    return ret;
}

int FimControlManager::ProgramCRFCode(void)
{
    std::cout << "fim::runtime::manager FimControlManager::ProgramCRFCode call" << std::endl;

    int ret = 0;

    /* ioctl by fim command */
    ret = fimDevice_->RequestIoctl();

    return ret;
}

int FimControlManager::SetFimMode(void)
{
    std::cout << "fim::runtime::manager FimControlManager::setFimMode call" << std::endl;

    int ret = 0;

    /* ioctl by fim command */
    ret = fimDevice_->RequestIoctl();

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
