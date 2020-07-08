#ifndef _FIM_EMULATOR_H_
#define _FIM_EMULATOR_H_

#include "FimTraceCoalescer.h"
#include "dramsim2/FimSimulator.h"
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
                                          int fmtd16_size, FimOpType op_type);
    int execute_bn(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size);
    int execute_elt_op(FimBo* output, FimBo* operand0, FimBo* operand1, FimMemTraceData* fmtd32, int fmtd32_size,
                       uint64_t fim_base_addr);
    int execute_relu(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, uint64_t fim_base_addr);
    int execute_gemv(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, FimOpType op_type,
                     uint64_t fim_base_addr, uint8_t* temp_buf);
    int execute_gemv_add(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, FimOpType op_type,
                         uint64_t fim_base_addr, uint8_t* temp_buf);

   private:
    FimBlockInfo fbi_;
    uint64_t fim_base_addr_;
    FimSimulator fim_sim_;
};

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_EMULATOR_H_ */
