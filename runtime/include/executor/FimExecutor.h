#ifndef _FIM_EXECUTOR_H_
#define _FIM_EXECUTOR_H_

#include "fim_data_types.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

namespace fim
{
namespace runtime
{
namespace executor
{
class FimExecutor
{
   public:
    FimExecutor(FimRuntimeType rtType, FimPrecision precision);
    virtual ~FimExecutor(void) {}

    static FimExecutor* getInstance(FimRuntimeType rtType, FimPrecision precision);

    int Initialize(void);
    int Deinitialize(void);
    int Execute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType);
    int Execute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType opType);

   private:
    FimRuntimeType rtType_;
    FimPrecision precision_;
    hipDeviceProp_t devProp_;
    size_t threadCnt_;
};

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_EXECUTOR_H_ */
