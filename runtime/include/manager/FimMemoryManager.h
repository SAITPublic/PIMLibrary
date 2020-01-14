#ifndef FIM_MEMORY_MANAGER_H_
#define FIM_MEMORY_MANAGER_H_

#include "manager/FimManager.h"
#include "fim_data_types.h"

namespace fim {
namespace runtime {
namespace manager {

class FimDevice;

class FimMemoryManager {
public:
    FimMemoryManager(FimDevice* fimDevice, FimRuntimeType rtType, FimPrecision precision);
    virtual ~FimMemoryManager(void);

    int Initialize(void);
    int Deinitialize(void);
    int AllocMemory(float** ptr, size_t size, FimMemType memType);
    int FreeMemory(float* ptr, FimMemType memType);
    int CopyMemory(float* dst, float* src, size_t size, FimMemcpyType);
    int DataReplacement(float* data, size_t size, FimOpType);

private:
    FimDevice *fimDevice_;
    FimRuntimeType rtType_;
    FimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif
