#include "manager/FimManager.h"
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
FimManager::FimManager(FimRuntimeType rtType, FimPrecision precision) : rtType_(rtType), precision_(precision)
{
    DLOG(INFO) << "called";
    fimDevice_ = new FimDevice(precision_);
    fimControlManager_ = new FimControlManager(fimDevice_, rtType_, precision_);
    fimMemoryManager_ = new FimMemoryManager(fimDevice_, rtType_, precision_);
}

FimManager::~FimManager(void)
{
    DLOG(INFO) << "called";
    delete fimDevice_;
    delete fimControlManager_;
    delete fimMemoryManager_;
}

FimManager* FimManager::getInstance(FimRuntimeType rtType, FimPrecision precision)
{
    DLOG(INFO) << "called";
    static FimManager* instance_ = new FimManager(rtType, precision);

    return instance_;
}

int FimManager::Initialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    fimDevice_->Initialize();
    fimControlManager_->Initialize();
    fimMemoryManager_->Initialize();

    return ret;
}

int FimManager::Deinitialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    fimMemoryManager_->Deinitialize();
    fimControlManager_->Deinitialize();
    fimDevice_->Deinitialize();

    return ret;
}

int FimManager::AllocMemory(void** ptr, size_t size, FimMemType memType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimMemoryManager_->AllocMemory(ptr, size, memType);

    return ret;
}

int FimManager::AllocMemory(FimBo* fimBo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimMemoryManager_->AllocMemory(fimBo);

    return ret;
}

int FimManager::FreeMemory(void* ptr, FimMemType memType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimMemoryManager_->FreeMemory(ptr, memType);

    return ret;
}

int FimManager::FreeMemory(FimBo* fimBo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimMemoryManager_->FreeMemory(fimBo);

    return ret;
}

int FimManager::CopyMemory(void* dst, void* src, size_t size, FimMemcpyType cpyType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimMemoryManager_->CopyMemory(dst, src, size, cpyType);

    return ret;
}

int FimManager::CopyMemory(FimBo* dst, FimBo* src, FimMemcpyType cpyType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimMemoryManager_->CopyMemory(dst, src, cpyType);

    return ret;
}

int FimManager::ConvertDataLayout(void* dst, void* src, size_t size, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimMemoryManager_->ConvertDataLayout(dst, src, size, opType);

    return ret;
}

int FimManager::ConvertDataLayout(FimBo* dst, FimBo* src, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimMemoryManager_->ConvertDataLayout(dst, src, opType);

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
