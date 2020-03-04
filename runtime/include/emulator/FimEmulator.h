#ifndef _FIM_EMULATOR_H_
#define _FIM_EMULATOR_H_

#include "fim_data_types.h"

namespace fim
{
namespace runtime
{
namespace emulator
{
class FimEmulator
{
   public:
    FimEmulator();
    virtual ~FimEmulator(void) {}

    static FimEmulator* get_instance(void);

    int initialize(void);
    int deinitialize(void);
    int convert_mem_trace_from_16B_to_32B(FimMemTraceData* fmtd32, int* fmtd32_size, FimMemTraceData* fmtd16,
                                          int fmtd16_size);
    int execute_fim(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, FimOpType op_type);

   private:
    FimBlockInfo fbi_;
    uint64_t fim_base_addr_;
};

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_EMULATOR_H_ */
