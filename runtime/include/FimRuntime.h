#ifndef _FIM_RUNTIME_H_
#define _FIM_RUNTIME_H_

#include "executor/FimExecutor.h"
#include "fim_data_types.h"
#include "manager/FimManager.h"

namespace fim
{
namespace runtime
{
class FimRuntime
{
   public:
    FimRuntime(FimRuntimeType rtType, FimPrecision precision);
    virtual ~FimRuntime(void) {}

    int Initialize(void);
    int Deinitialize(void);
    int AllocMemory(void** ptr, size_t size, FimMemType memType);
    int FreeMemory(void* ptr, FimMemType memType);
    int ConvertDataLayout(void* dst, void* src, size_t size, FimOpType opType);
    int CopyMemory(void* dst, void* src, size_t size, FimMemcpyType);
    int Execute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType);

   private:
    fim::runtime::manager::FimManager* fimManager_;
    fim::runtime::executor::FimExecutor* fimExecutor_;
    FimRuntimeType rtType_;
    FimPrecision precision_;
};

} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_RUNTIME_H_ */
