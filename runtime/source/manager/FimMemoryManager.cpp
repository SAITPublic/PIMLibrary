#include "manager/FimMemoryManager.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"
#include "utility/fim_util.h"

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
    get_fim_block_info(&fbi_);
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

int FimMemoryManager::ConvertDataLayout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType opType)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (opType == OP_ELT_ADD) {
        convertDataLayoutForEltAdd(dst, src0, FimBankType::EVEN_BANK);
        convertDataLayoutForEltAdd(dst, src1, FimBankType::ODD_BANK);
    } else {
        DLOG(ERROR) << "not yet implemented";
    }

    return ret;
}

#include "executor/fim_hip_kernels/fim_op_kernels.fimk"
int FimMemoryManager::convertDataLayoutForEltAdd(FimBo* dst, FimBo* src, FimBankType fimBankType)
{
    DLOG(INFO) << "called";
    uint32_t ret = 0;

    uint32_t cidx = 0;
    uint32_t rank = 0;
    uint32_t bg = 0;
    uint32_t bank = 0;
    uint32_t s_row = 0;
    uint32_t s_col = 0;
    uint32_t trans_size = 32;
    uint32_t type_size = (precision_ == FIM_FP16) ? sizeof(FP16) : sizeof(char);
    uint64_t addr_op = 0x0;
    uint32_t dim_operand = dst->size / trans_size / type_size;
    char* dst_data = (char*)dst->data;
    char* src_data = (char*)src->data;
    int num_grf = fbi_.num_grf;
    int num_banks = fbi_.num_banks;
    int num_fim_blocks = fbi_.num_fim_blocks;
    int num_bank_groups = fbi_.num_bank_groups;
    int num_fim_rank = fbi_.num_fim_rank;
    int num_fim_chan = fbi_.num_fim_chan;

    for (int x = 0; x < dim_operand; x += num_grf) {
        uint32_t row = 0;
        uint32_t col = 0;

        for (int grf_idx = 0; grf_idx < num_grf; grf_idx++) {
            addr_op = addr_gen_safe(cidx, rank, bg, bank + (int)fimBankType, row, col);
            memcpy(dst_data + addr_op, src_data + (x + grf_idx) * trans_size, trans_size);
            col++;
        }

        bank += (num_banks / num_fim_blocks);

        if (bank >= (num_banks / num_bank_groups)) {
            bg++;
            bank = 0;
        }

        if (bg >= num_bank_groups) {
            bg = 0;
            rank++;
        }

        if (rank >= num_fim_rank) {
            rank = 0;
            cidx++;
        }

        if (cidx >= num_fim_chan) {
            cidx = 0;
            s_row = row;
            s_col = col;
        }
    }

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
