#ifndef _FIM_EXECUTOR_H_
#define _FIM_EXECUTOR_H_

#include "emulator/FimEmulator.h"
#include "fim_data_types.h"
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "manager/FimManager.h"

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

    int execute_add(FimBo* output, FimBo* operand0, FimBo* operand1);
    int execute_mul(FimBo* output, FimBo* operand0, FimBo* operand1);
    int execute_relu(FimBo* output, FimBo* fim_data);
    int execute_gemv(FimBo* output, FimBo* operand0, FimBo* operand1);
    int execute_bn(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance,
                   double epsilon);

    int preprocess_srf(FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance, double epsilon, uint8_t* srf_binary);

   private:
    fim::runtime::manager::FimManager* fim_manager_;
    uint8_t* d_crf_bin_buffer_;
    uint8_t* d_srf_bin_buffer_;

    FimRuntimeType rt_type_;
    FimPrecision precision_;
    hipDeviceProp_t dev_prop_;
    size_t thread_cnt_;
    uint64_t fim_base_addr_;
    uint8_t* fim_gemv_tmp_buffer_;
    FimBlockInfo fbi_;
#ifdef EMULATOR
    FimMemTraceData* d_fmtd16_;
    int* d_fmtd16_size_;
    fim::runtime::emulator::FimEmulator* fim_emulator_;
    FimMemTraceData* h_fmtd16_;
    FimMemTraceData* h_fmtd32_;
    int* h_fmtd16_size_;
    int* h_fmtd32_size_;
    int fmtd_size_per_ch_;
    int max_block_size_;
    int max_fmtd_size_;
#endif
};

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_EXECUTOR_H_ */
