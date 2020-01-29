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
    int FreeMemory(void* ptr, FimMemType memType);
    int CopyMemory(void* dst, void* src, size_t size, FimMemcpyType);
    int ConvertDataLayout(void* dst, void* src, size_t size, FimOpType);

   private:
    FimDevice* fimDevice_;
    FimRuntimeType rtType_;
    FimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_MEMORY_MANAGER_H_ */
