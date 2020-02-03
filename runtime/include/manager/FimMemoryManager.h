#ifndef _FIM_MEMORY_MANAGER_H_
#define _FIM_MEMORY_MANAGER_H_

#include "fim_data_types.h"
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
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_MEMORY_MANAGER_H_ */
