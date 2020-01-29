#include "manager/FimMemoryManager.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimMemoryManager::FimMemoryManager(FimDevice* fimDevice, FimRuntimeType rtType, FimPrecision precision)
    : fimDevice_(fimDevice), rtType_(rtType), precision_(precision)
{
    std::cout << "fim::runtime::manager FimMemoryManager creator call" << std::endl;
}

FimMemoryManager::~FimMemoryManager(void)
{
    std::cout << "fim::runtime::manager FimMemoryManager destroyer call" << std::endl;
}

int FimMemoryManager::Initialize(void)
{
    std::cout << "fim::runtime::manager FimMemoryManager::Initialize call" << std::endl;

    int ret = 0;

    return ret;
}

int FimMemoryManager::Deinitialize(void)
{
    std::cout << "fim::runtime::manager FimMemoryManager::Deinitialize call" << std::endl;

    int ret = 0;

    return ret;
}

int FimMemoryManager::AllocMemory(void** ptr, size_t size, FimMemType memType)
{
    std::cout << "fim::runtime::manager FimMemoryManager::AllocMemory call" << std::endl;

    int ret = 0;

    if (memType == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)ptr, size) != hipSuccess) {
            return -1;
        }
    } else if (memType == MEM_TYPE_HOST) {
        *ptr = (void*)malloc(size);
    } else if (memType == MEM_TYPE_FIM) {
        /* todo:implement fimalloc function */
        if (hipMalloc((void**)ptr, size) != hipSuccess) {
            return -1;
        }
    }

    return ret;
}

int FimMemoryManager::FreeMemory(void* ptr, FimMemType memType)
{
    std::cout << "fim::runtime::manager FimMemoryManager::FreeMemory call" << std::endl;

    int ret = 0;

    if (memType == MEM_TYPE_DEVICE) {
        if (hipFree(ptr) != hipSuccess) {
            return -1;
        }
    } else if (memType == MEM_TYPE_HOST) {
        free(ptr);
    } else if (memType == MEM_TYPE_FIM) {
        /* todo:implement fimfree function */
        if (hipFree(ptr) != hipSuccess) {
            return -1;
        }
    }

    return ret;
}

int FimMemoryManager::CopyMemory(void* dst, void* src, size_t size, FimMemcpyType cpyType)
{
    std::cout << "fim::runtime::manager FimMemoryManager::CopyMemory call" << std::endl;

    int ret = 0;

    if (cpyType == HOST_TO_FIM || cpyType == HOST_TO_DEVICE) {
        if (hipMemcpy(dst, src, size, hipMemcpyHostToDevice) != hipSuccess) {
            return -1;
        }
    } else if (cpyType == FIM_TO_HOST || cpyType == DEVICE_TO_HOST) {
        if (hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) != hipSuccess) {
            return -1;
        }
    } else if (cpyType == DEVICE_TO_FIM || cpyType == FIM_TO_DEVICE) {
        if (hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) != hipSuccess) {
            return -1;
        }
    } else if (cpyType == HOST_TO_HOST) {
        if (hipMemcpy(dst, src, size, hipMemcpyHostToHost) != hipSuccess) {
            return -1;
        }
    }

    return ret;
}

int FimMemoryManager::ConvertDataLayout(void* dst, void* src, size_t size, FimOpType opType)
{
    std::cout << "fim::runtime::manager FimMemoryManager::ConvertDataLayout call" << std::endl;

    int ret = 0;

    /* todo: implement ConvertDataLayout function refer to memory map */
    hipMemcpy(dst, src, size, hipMemcpyHostToDevice);

    return ret;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
