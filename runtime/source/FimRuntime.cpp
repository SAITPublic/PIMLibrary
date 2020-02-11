#include "FimRuntime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "executor/FimExecutor.h"
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
FimRuntime::FimRuntime(FimRuntimeType rtType, FimPrecision precision) : rtType_(rtType), precision_(precision)
{
    DLOG(INFO) << "called";

    fimManager_ = fim::runtime::manager::FimManager::getInstance(rtType, precision);
    fimExecutor_ = fim::runtime::executor::FimExecutor::getInstance(rtType, precision);
}

int FimRuntime::Initialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    fimManager_->Initialize();
    fimExecutor_->Initialize();

    return ret;
}

int FimRuntime::Deinitialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

int FimRuntime::AllocMemory(void** ptr, size_t size, FimMemType memType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->AllocMemory(ptr, size, memType);

    return ret;
}

int FimRuntime::AllocMemory(FimBo* fimBo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->AllocMemory(fimBo);

    return ret;
}

int FimRuntime::FreeMemory(void* ptr, FimMemType memType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->FreeMemory(ptr, memType);

    return ret;
}

int FimRuntime::FreeMemory(FimBo* fimBo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->FreeMemory(fimBo);

    return ret;
}

int FimRuntime::ConvertDataLayout(void* dst, void* src, size_t size, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->ConvertDataLayout(dst, src, size, opType);

    return ret;
}

int FimRuntime::ConvertDataLayout(FimBo* dst, FimBo* src, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->ConvertDataLayout(dst, src, opType);

    return ret;
}

int FimRuntime::ConvertDataLayout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->ConvertDataLayout(dst, src0, src1, opType);

    return ret;
}

int FimRuntime::CopyMemory(void* dst, void* src, size_t size, FimMemcpyType cpyType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->CopyMemory(dst, src, size, cpyType);

    return ret;
}

int FimRuntime::CopyMemory(FimBo* dst, FimBo* src, FimMemcpyType cpyType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimManager_->CopyMemory(dst, src, cpyType);

    return ret;
}

int FimRuntime::Execute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimExecutor_->Execute(output, operand0, operand1, size, opType);

    return ret;
}

int FimRuntime::Execute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimExecutor_->Execute(output, operand0, operand1, opType);

    return ret;
}

int FimRuntime::Execute(FimBo* output, FimBo* fimData, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fimExecutor_->Execute(output, fimData, opType);

    return ret;
}

} /* namespace runtime */
} /* namespace fim */
