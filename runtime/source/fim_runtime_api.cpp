#include <iostream>
#include "hip/hip_runtime.h"
#include "fim_runtime_api.h"
#include "FimRuntime.h"

using namespace fim::runtime;

FimRuntime* fimRuntime = nullptr;

int FimInitialize(FimRuntimeType rtType)
{
    int ret = 0;
    std::cout << "fim::runtime::api Initialize call" << std::endl;

    if (fimRuntime == nullptr)
        fimRuntime = new FimRuntime(rtType);

    ret = fimRuntime->Initialize();

    return ret;
}

int FimDeinitialize(void)
{
    std::cout << "fim::runtime::api Deinitialize call" << std::endl;

    int ret = 0;
    if (fimRuntime != nullptr) {
        ret = fimRuntime->Deinitialize();
        delete fimRuntime;
        fimRuntime = nullptr;
    }

    return ret;
}

int FimAllocMemory(float** ptr, size_t size, FimMemType memType)
{
    std::cout << "fim::runtime::api FimAllocMemory call" << std::endl;

    int ret = 0;
    if (fimRuntime == nullptr) {
        return -1;
    }

    ret = fimRuntime->AllocMemory(ptr, size, memType);
    if (ret < 0) {
        return -1;
    }

    return ret;
}

int FimFreeMemory(float* ptr, FimMemType memType)
{
    std::cout << "fim::runtime::api FimFreeMemory call" << std::endl;

    int ret;
    if (fimRuntime == nullptr) {
        std::cout << "fim::runtime::api Fail to FimFreeMemory" << std::endl;
        return -1;
    }
    ret = fimRuntime->FreeMemory(ptr, memType);
    return ret;
}

int FimCopyMemory(float* dst, float* src, size_t size, FimMemcpyType cpyType)
{
    std::cout << "fim::runtime::api FimAllocMemory call" << std::endl;

    int ret = 0;
    if (fimRuntime == nullptr) {
        return -1;
    }

    ret = fimRuntime->CopyMemory(dst, src, size, cpyType);
    if (ret < 0) {
        return -1;
    }

    return ret;
}

int FimExecute(float* output, float* operand0, float* operand1, size_t size, FimOpType opType, FimPrecision precision)
{
    std::cout << "fim::runtime::api FimExecute call" << std::endl;

    int ret = 0;
    if (fimRuntime == nullptr) {
        return -1;
    }

    ret = fimRuntime->Execute(output, operand0, operand1, size, opType, precision);
    if (ret < 0) {
        return -1;
    }

    return ret;
}

