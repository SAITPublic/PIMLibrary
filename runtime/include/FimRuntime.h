#ifndef FIM_RUNTIME_H_
#define FIM_RUNTIME_H_

#include "manager/FimManager.h"
#include "executor/FimExecutor.h"
#include "fim_data_types.h"

namespace fim {
namespace runtime {

class FimRuntime {
public:
    FimRuntime(FimRuntimeType rtType, FimPrecision precision);
    virtual ~FimRuntime(void) {}

    int Initialize(void);
    int Deinitialize(void);
    int AllocMemory(float** ptr, size_t size, FimMemType memType);
    int FreeMemory(float* ptr, FimMemType memType);
    int DataReplacement(float* data, size_t size, FimOpType opType);
    int CopyMemory(float* dst, float* src, size_t size, FimMemcpyType);
    int Execute(float* output, float* operand0, float* operand1, size_t size, FimOpType opType);

private:
    fim::runtime::manager::FimManager* fimManager_;
    fim::runtime::executor::FimExecutor* fimExecutor_;
    FimRuntimeType rtType_;
    FimPrecision precision_;
    hipDeviceProp_t devProp_;
};

} /* namespace runtime */
} /* namespace fim */

#endif
