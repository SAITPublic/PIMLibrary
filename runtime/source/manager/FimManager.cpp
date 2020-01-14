#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "manager/FimManager.h"

namespace fim {
namespace runtime {
namespace manager {

FimManager::FimManager(FimRuntimeType rtType, FimPrecision precision)
    : rtType_(rtType), precision_(precision)
{
    std::cout << "fim::runtime::manager creator call" << std::endl;

    fimDevice_  = new FimDevice(precision_);
    fimControlManager_ = new FimControlManager(fimDevice_, rtType_, precision_);
    fimMemoryManager_  = new FimMemoryManager(fimDevice_, rtType_, precision_);
}

FimManager::~FimManager(void)
{
    delete fimDevice_;
    delete fimControlManager_;
    delete fimMemoryManager_;
}

FimManager* FimManager::getInstance(FimRuntimeType rtType, FimPrecision precision)
{
    std::cout << "fim::runtime::manager getInstance call" << std::endl;

    static FimManager* instance_ = new FimManager(rtType, precision);

    return instance_;
}

int FimManager::Initialize(void)
{
    std::cout << "fim::runtime::manager Initialize call" << std::endl;

    int ret = 0;

    fimDevice_->Initialize();
    fimControlManager_->Initialize();
    fimMemoryManager_->Initialize();

    return ret;
}

int FimManager::Deinitialize(void)
{
    std::cout << "fim::runtime::manager Deinitialize call" << std::endl;

    int ret = 0;

    fimMemoryManager_->Deinitialize();
    fimControlManager_->Deinitialize();
    fimDevice_->Deinitialize();

    return ret;
}

int FimManager::AllocMemory(float** ptr, size_t size, FimMemType memType)
{
    std::cout << "fim::runtime::manager AllocMemory call" << std::endl;

    int ret = 0;

    ret = fimMemoryManager_->AllocMemory(ptr, size, memType);

    return ret;
}

int FimManager::FreeMemory(float* ptr, FimMemType memType)
{
    std::cout << "fim::runtime::manager FreeMemory call" << std::endl;

    int ret = 0;

    ret = fimMemoryManager_->FreeMemory(ptr, memType);

    return ret;
}

int FimManager::CopyMemory(float* dst, float* src, size_t size, FimMemcpyType cpyType)
{
    std::cout << "fim::runtime::manager CopyeMemory call" << std::endl;

    int ret = 0;

    ret = fimMemoryManager_->CopyMemory(dst, src, size, cpyType);

    return ret;
}

int FimManager::DataReplacement(float* data, size_t size, FimOpType opType)
{
    std::cout << "fim::runtime::manager CopyeMemory call" << std::endl;

    int ret = 0;

    ret = fimMemoryManager_->DataReplacement(data, size, opType);

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
