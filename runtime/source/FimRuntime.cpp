#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "executor/FimExecutor.h"
#include "FimRuntime.h"

namespace fim {
namespace runtime {

FimRuntime::FimRuntime(FimRuntimeType rtType, FimPrecision precision)
    :rtType_(rtType), precision_(precision)
{
    std::cout << "fim::runtime::FimRuntime creator call" << std::endl;

    fimManager_ = fim::runtime::manager::FimManager::getInstance(rtType, precision);
    fimExecutor_ = fim::runtime::executor::FimExecutor::getInstance(rtType, precision);
}

int FimRuntime::Initialize(void)
{
    std::cout << "fim::runtime Initialize call" << std::endl;

    int ret = 0;

    return ret;
}

int FimRuntime::Deinitialize(void)
{
    std::cout << "fim::runtime Deinitialize call" << std::endl;

    int ret = 0;

    return ret;
}

int FimRuntime::AllocMemory(void** ptr, size_t size, FimMemType memType)
{
    std::cout << "fim::runtime AllocMemory call" << std::endl;

    int ret = 0;

    ret = fimManager_->AllocMemory(ptr, size, memType);

    return ret;
}

int FimRuntime::FreeMemory(void* ptr, FimMemType memType)
{
    std::cout << "fim::runtime FreeMemory call" << std::endl;

    int ret = 0;

    ret = fimManager_->FreeMemory(ptr, memType);

    return ret;
}

int FimRuntime::DataReplacement(void* data, size_t size, FimOpType opType)
{
    std::cout << "fim::runtime DataReplacement call" << std::endl;

    int ret = 0;

    ret = fimManager_->DataReplacement(data, size, opType);

    return ret;
}

int FimRuntime::CopyMemory(void* dst, void* src, size_t size, FimMemcpyType cpyType)
{
    std::cout << "fim::runtime Memcpy call" << std::endl;

    int ret = 0;

    ret = fimManager_->CopyMemory(dst, src, size, cpyType);

    return ret;
}

int FimRuntime::Execute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType)
{
    std::cout << "fim::runtime Execute call" << std::endl;

    int ret = 0;

    ret = fimExecutor_->Execute(output, operand0, operand1, size, opType);

    return ret;
}

} /* namespace runtime */
} /* namespace fim */

