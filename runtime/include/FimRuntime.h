#ifndef FIM_RUNTIME_H_
#define FIM_RUNTIME_H_

#include "fim_runtime_api.h"
#include "hip/hip_runtime.h"

namespace fim {
namespace runtime {

class FimRuntime {
public:
    FimRuntime(FimRuntimeType rtType);
    virtual ~FimRuntime(void) {}

    int Initialize(void);
    int Deinitialize(void);
    int AllocMemory(float** ptr, size_t size, FimMemType memType);
    int FreeMemory(float* ptr, FimMemType memType);
    int CopyMemory(float* dst, float* src, size_t size, FimMemcpyType);
    int Execute(float* output, float* operand0, float* operand1, size_t size, FimOpType opType, FimPrecision precision);

private:
    int fim_fd_;
    static constexpr char fimDrvName_[] = "/dev/fim_drv";
    size_t threadCnt_;
    size_t blockCnt_;
    FimRuntimeType rtType_;
	hipDeviceProp_t devProp_;
};

} /* namespace runtime */
} /* namespace fim */

#endif
