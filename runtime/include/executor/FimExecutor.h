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
    FimExecutor(FimRuntimeType rt_type, FimPrecision precision);
    virtual ~FimExecutor(void) {}

    static FimExecutor* get_instance(FimRuntimeType rt_type, FimPrecision precision);

    int initialize(void);
    int deinitialize(void);
    int execute(void* output, void* operand0, void* operand1, size_t size, FimOpType op_type);
    int execute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType op_type);
    int execute(FimBo* output, FimBo* fim_data, FimOpType op_type);

   private:
    FimRuntimeType rt_type_;
    FimPrecision precision_;
    FimBlockInfo fbi_;
    hipDeviceProp_t dev_prop_;
    size_t thread_cnt_;
    uint64_t fim_base_addr_;
};

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_EXECUTOR_H_ */
