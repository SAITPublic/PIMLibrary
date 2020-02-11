#ifndef _FIM_MEMORY_MANAGER_H_
#define _FIM_MEMORY_MANAGER_H_

#include "fim_data_types.h"
#include "internal/simple_heap.hpp"
#include "manager/FimManager.h"

namespace fim
{
namespace runtime
{
namespace manager
{
class FimDevice;

class FimMemoryManager
{
   public:
    FimMemoryManager(FimDevice* fimDevice, FimRuntimeType rtType, FimPrecision precision);
    virtual ~FimMemoryManager(void);

    int Initialize(void);
    int Deinitialize(void);
    int AllocMemory(void** ptr, size_t size, FimMemType memType);
    int AllocMemory(FimBo* fimBo);
    int FreeMemory(void* ptr, FimMemType memType);
    int FreeMemory(FimBo* fimBo);
    int CopyMemory(void* dst, void* src, size_t size, FimMemcpyType);
    int CopyMemory(FimBo* dst, FimBo* src, FimMemcpyType);
    int ConvertDataLayout(void* dst, void* src, size_t size, FimOpType);
    int ConvertDataLayout(FimBo* dst, FimBo* src, FimOpType);
    int ConvertDataLayout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType);

   private:
    int convertDataLayoutForEltAdd(FimBo* dst, FimBo* src, FimBankType fimBankType);
    uint32_t mask_by_bit(uint32_t value, uint32_t start, uint32_t end);
    uint64_t addr_gen(uint32_t chan, uint32_t rank, uint32_t bankgroup, uint32_t bank, uint32_t row, uint32_t col);
    uint64_t addr_gen_safe(uint32_t chan, uint32_t rank, uint32_t bg, uint32_t bank, uint32_t& row, uint32_t& col);

   private:
    FimDevice* fimDevice_;
    FimRuntimeType rtType_;
    FimPrecision precision_;

    /* TODO: get VEGA20 scheme from device driver */
    FimAddrMap fim_addr_map_ = AMDGPU_VEGA20;
    const int num_banks_ = 16;
    const int num_fim_blocks_ = 8;
    const int num_bank_groups_ = 4;
    const int num_fim_rank_ = 1;
    const int num_fim_chan_ = 64;
    const int num_rank_bit_ = 1;
    const int num_row_bit_ = 14;
    const int num_col_high_bit_ = 3;
    const int num_bank_high_bit_ = 1;
    const int num_bankgroup_bit_ = 2;
    const int num_bank_low_bit_ = 1;
    const int num_chan_bit_ = 6;
    const int num_col_low_bit_ = 2;
    const int num_offset_bit_ = 5;
    const int num_grf_ = 8;
    const int num_col_ = 128;
    const int num_row_ = 16384;
    const int bl_ = 4;

    /**
     * @brief FIM Block allocator of size 2MB
     *
     * TODO: This is a simple block allocator where it uses malloc for allocation and free
     *       It has to be modified to use FIM memory region for alloc and free.
     */

    class FimBlockAllocator
    {
       private:
        static const size_t block_size_ = 2 * 1024 * 1024;  // 2MB blocks.

       public:
        explicit FimBlockAllocator() {}
        void* alloc(size_t request_size, size_t& allocated_size) const;
        void free(void* ptr, size_t length) const;
        size_t block_size() const { return block_size_; }
    };

    SimpleHeap<FimBlockAllocator> fragment_allocator_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_MEMORY_MANAGER_H_ */
