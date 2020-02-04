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
FimControlManager::FimControlManager(FimDevice* fimDevice, FimRuntimeType rtType, FimPrecision precision)
    : fimDevice_(fimDevice), rtType_(rtType), precision_(precision)
{
    LOGI(FIM_CON_MG, "called");
}

FimControlManager::~FimControlManager(void) { LOGI(FIM_CON_MG, "called"); }

int FimControlManager::Initialize(void)
{
    LOGI(FIM_CON_MG, "called");

    int ret = 0;

    return ret;
}

int FimControlManager::Deinitialize(void)
{
    LOGI(FIM_CON_MG, "called");

    int ret = 0;

    return ret;
}

int FimControlManager::ProgramCRFCode(void)
{
    LOGI(FIM_CON_MG, "called");

    int ret = 0;

    /* ioctl by fim command */
    ret = fimDevice_->RequestIoctl();

    return ret;
}

int FimControlManager::SetFimMode(void)
{
    LOGI(FIM_CON_MG, "called");

    int ret = 0;

    /* ioctl by fim command */
    ret = fimDevice_->RequestIoctl();

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
