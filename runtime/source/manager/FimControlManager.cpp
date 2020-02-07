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
    DLOG(INFO) << "called";
}

FimControlManager::~FimControlManager(void) { DLOG(INFO) << "called"; }
int FimControlManager::Initialize(void)
{
    DLOG(INFO) << "called";

    int ret = 0;

    return ret;
}

int FimControlManager::Deinitialize(void)
{
    DLOG(INFO) << "called";

    int ret = 0;

    return ret;
}

int FimControlManager::ProgramCRFCode(void)
{
    DLOG(INFO) << "called";

    int ret = 0;

    /* ioctl by fim command */
    ret = fimDevice_->RequestIoctl();

    return ret;
}

int FimControlManager::SetFimMode(void)
{
    DLOG(INFO) << "called";

    int ret = 0;

    /* ioctl by fim command */
    ret = fimDevice_->RequestIoctl();

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
