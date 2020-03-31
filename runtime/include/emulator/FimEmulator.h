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
    int execute_fim(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, FimOpType op_type);

    int set_input_for_test(uint16_t* test_input, int test_size);
    int set_kernel_trace_for_test(FimMemTraceData* fmtd32, int* num_trace);
    bool uint16_equal(uint16_t A, uint16_t B, int maxUlpsDiff);
    int compare_output(uint16_t* test_output, FimBo* output);

   private:
    FimBlockInfo fbi_;
    uint64_t fim_base_addr_;
    FimSimulator fim_sim_;
};

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_EMULATOR_H_ */
