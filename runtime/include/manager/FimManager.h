#ifndef _FIM_MANAGER_H_
#define _FIM_MANAGER_H_

#include "fim_data_types.h"
#include "manager/FimControlManager.h"
#include "manager/FimDevice.h"
#include "manager/FimMemoryManager.h"

namespace fim
{
namespace runtime
{
namespace manager
{
class FimMemoryManager;
class FimControlManager;
class FimDevice;

class FimManager
{
   public:
    virtual ~FimManager(void);

    static FimManager* getInstance(FimRuntimeType rtType, FimPrecision precision);

    int Initialize(void);
    int Deinitialize(void);
    int AllocMemory(void** ptr, size_t size, FimMemType memType);
    int FreeMemory(void* ptr, FimMemType memType);
    int CopyMemory(void* dst, void* src, size_t size, FimMemcpyType);
    int ConvertDataLayout(void* dst, void* src, size_t size, FimOpType);

   private:
    FimManager(FimRuntimeType rtType, FimPrecision precision);

    FimDevice* fimDevice_;
    FimControlManager* fimControlManager_;
    FimMemoryManager* fimMemoryManager_;

    FimRuntimeType rtType_;
    FimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_MANAGER_H_ */
