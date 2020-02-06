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

   private:
    FimDevice* fimDevice_;
    FimRuntimeType rtType_;
    FimPrecision precision_;

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
