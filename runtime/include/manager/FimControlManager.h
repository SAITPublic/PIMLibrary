#ifndef _FIM_CONTROL_MANAGER_H_
#define _FIM_CONTROL_MANAGER_H_

#include "fim_data_types.h"
#include "manager/FimManager.h"

namespace fim
{
namespace runtime
{
namespace manager
{
class FimDevice;

class FimControlManager
{
   public:
    FimControlManager(FimDevice* fimDevice, FimRuntimeType rtType, FimPrecision precision);
    virtual ~FimControlManager(void);

    int Initialize(void);
    int Deinitialize(void);
    int ProgramCRFCode(void);
    int SetFimMode(void);

   private:
    FimDevice* fimDevice_;
    FimRuntimeType rtType_;
    FimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_CONTROL_MANAGER_H_ */
