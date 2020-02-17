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
    FimControlManager(FimDevice* fim_device, FimRuntimeType rt_type, FimPrecision precision);
    virtual ~FimControlManager(void);

    int initialize(void);
    int deinitialize(void);
    int program_crf_code(void);
    int set_fim_mode(void);

   private:
    FimDevice* fim_device_;
    FimRuntimeType rt_type_;
    FimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_CONTROL_MANAGER_H_ */
