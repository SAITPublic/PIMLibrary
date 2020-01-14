#ifndef FIM_MANAGER_H_
#define FIM_MANAGER_H_

#include "manager/FimControlManager.h"
#include "manager/FimMemoryManager.h"
#include "manager/FimDevice.h"
#include "fim_data_types.h"

namespace fim {
namespace runtime {
namespace manager {

class FimMemoryManager;
class FimControlManager;
class FimDevice;

class FimManager {
public:
    virtual ~FimManager(void);

    static FimManager* getInstance(FimRuntimeType rtType, FimPrecision precision);

    int Initialize(void);
    int Deinitialize(void);
    int AllocMemory(float** ptr, size_t size, FimMemType memType);
	int FreeMemory(float* ptr, FimMemType memType);
    int CopyMemory(float* dst, float* src, size_t size, FimMemcpyType);
	int DataReplacement(float* data, size_t size, FimOpType);

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



#endif
