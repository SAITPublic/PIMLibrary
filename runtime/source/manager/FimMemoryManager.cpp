#include "manager/FimMemoryManager.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimMemoryManager::FimMemoryManager(FimDevice* fimDevice, FimRuntimeType rtType, FimPrecision precision)
    : fimDevice_(fimDevice), rtType_(rtType), precision_(precision)
{
    DLOG(INFO) << "called";
}

FimMemoryManager::~FimMemoryManager(void) { DLOG(INFO) << "called"; }
int FimMemoryManager::Initialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

int FimMemoryManager::Deinitialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

int FimMemoryManager::AllocMemory(void** ptr, size_t size, FimMemType memType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (memType == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)ptr, size) != hipSuccess) {
            return -1;
        }
    } else if (memType == MEM_TYPE_HOST) {
        *ptr = (void*)malloc(size);
    } else if (memType == MEM_TYPE_FIM) {
        *ptr = fragment_allocator_.alloc(size);
    }

    return ret;
}

int FimMemoryManager::AllocMemory(FimBo* fimBo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (fimBo->memType == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)&fimBo->data, fimBo->size) != hipSuccess) {
            return -1;
        }
    } else if (fimBo->memType == MEM_TYPE_HOST) {
        fimBo->data = (void*)malloc(fimBo->size);
    } else if (fimBo->memType == MEM_TYPE_FIM) {
        fimBo->data = fragment_allocator_.alloc(fimBo->size);
    }

    return ret;
}

int FimMemoryManager::FreeMemory(void* ptr, FimMemType memType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (memType == MEM_TYPE_DEVICE) {
        if (hipFree(ptr) != hipSuccess) {
            return -1;
        }
    } else if (memType == MEM_TYPE_HOST) {
        free(ptr);
    } else if (memType == MEM_TYPE_FIM) {
        return fragment_allocator_.free(ptr);
    }

    return ret;
}

int FimMemoryManager::FreeMemory(FimBo* fimBo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (fimBo->memType == MEM_TYPE_DEVICE) {
        if (hipFree(fimBo->data) != hipSuccess) {
            return -1;
        }
    } else if (fimBo->memType == MEM_TYPE_HOST) {
        free(fimBo->data);
    } else if (fimBo->memType == MEM_TYPE_FIM) {
        if (fragment_allocator_.free(fimBo->data)) return 0;
        return -1;
    }

    return ret;
}

int FimMemoryManager::CopyMemory(void* dst, void* src, size_t size, FimMemcpyType cpyType)
{
    DLOG(INFO) << "called";
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

int FimMemoryManager::CopyMemory(FimBo* dst, FimBo* src, FimMemcpyType cpyType)
{
    DLOG(INFO) << "called";
    int ret = 0;
    size_t size = dst->size;

    if (cpyType == HOST_TO_FIM || cpyType == HOST_TO_DEVICE) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyHostToDevice) != hipSuccess) {
            return -1;
        }
    } else if (cpyType == FIM_TO_HOST || cpyType == DEVICE_TO_HOST) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyDeviceToHost) != hipSuccess) {
            return -1;
        }
    } else if (cpyType == DEVICE_TO_FIM || cpyType == FIM_TO_DEVICE) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyDeviceToDevice) != hipSuccess) {
            return -1;
        }
    } else if (cpyType == HOST_TO_HOST) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyHostToHost) != hipSuccess) {
            return -1;
        }
    }

    return ret;
}

int FimMemoryManager::ConvertDataLayout(void* dst, void* src, size_t size, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    /* todo: implement ConvertDataLayout function refer to memory map */
    hipMemcpy(dst, src, size, hipMemcpyHostToDevice);

    return ret;
}

int FimMemoryManager::ConvertDataLayout(FimBo* dst, FimBo* src, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;
    size_t size = dst->size;

    /* todo: implement ConvertDataLayout function refer to memory map */
    hipMemcpy(dst->data, src->data, size, hipMemcpyHostToDevice);

    return ret;
}

void* FimMemoryManager::FimBlockAllocator::alloc(size_t request_size, size_t& allocated_size) const
{
    assert(request_size <= block_size() && "BlockAllocator alloc request exceeds block size.");
    void* ret = nullptr;
    size_t bsize = block_size();

    /* todo:implement fimalloc function */
    ret = (void*)malloc(bsize);

    assert(ret != nullptr && "Region returned nullptr on success.");

    allocated_size = block_size();
    return ret;
}

void FimMemoryManager::FimBlockAllocator::free(void* ptr, size_t length) const
{
    if (ptr == NULL || length == 0) {
        return;
    }

    /* todo:implement fimfree function */
    std::free(ptr);
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
