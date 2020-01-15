#ifndef FIM_EXECUTOR_H_
#define FIM_EXECUTOR_H_

#include "hip/hip_runtime.h"
#include "fim_data_types.h"

namespace fim {
namespace runtime {
namespace executor {

class FimExecutor {
public:
    FimExecutor(FimRuntimeType rtType, FimPrecision precision);
    virtual ~FimExecutor(void) {}

    static FimExecutor* getInstance(FimRuntimeType rtType, FimPrecision precision);

    int Initialize(void);
    int Deinitialize(void);
    int Execute(float* output, float* operand0, float* operand1, size_t size, FimOpType opType);

private:
    FimRuntimeType rtType_;
    FimPrecision precision_;
    hipDeviceProp_t devProp_;
    size_t threadCnt_;
};

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */

#endif
