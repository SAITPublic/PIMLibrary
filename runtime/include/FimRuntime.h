#ifndef _FIM_RUNTIME_H_
#define _FIM_RUNTIME_H_

#include <unordered_map>
#include "executor/FimExecutor.h"
#include "fim_data_types.h"
#include "manager/FimManager.h"

namespace fim
{
namespace runtime
{
class FimRuntime
{
   public:
    FimRuntime(FimRuntimeType rtType, FimPrecision precision);
    virtual ~FimRuntime(void) {}

    int initialize(void);
    int deinitialize(void);
    int alloc_memory(void** ptr, size_t size, FimMemType mem_type);
    int alloc_memory(FimBo* fimBo);
    int free_memory(void* ptr, FimMemType mem_type);
    int free_memory(FimBo* fimBo);
    int convert_data_layout(void* dst, void* src, size_t size, FimOpType op_type);
    int convert_data_layout(FimBo* dst, FimBo* src, FimOpType op_type);
    int convert_data_layout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type);
    int copy_memory(void* dst, void* src, size_t size, FimMemCpyType cpy_type);
    int copy_memory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type);

    int execute_add(FimBo* output, FimBo* operand0, FimBo* operand1);
    int execute_mul(FimBo* output, FimBo* operand0, FimBo* operand1);
    int execute_relu(FimBo* output, FimBo* fim_data);
    int execute_gemv(FimBo* output, FimBo* operand0, FimBo* operand1);
    int execute_bn(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance,
                   double epsilon);
    int execute_dummy(void);
    int insert_weight(uint64_t w_addr, void* fim_addr);
    void* find_weight(uint64_t w_addr);

   private:
    fim::runtime::manager::FimManager* fim_manager_;
    fim::runtime::executor::FimExecutor* fim_executor_;
    FimRuntimeType rt_type_;
    FimPrecision precision_;
    std::unordered_map<uint64_t, void*> weight_map_;
};

} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_RUNTIME_H_ */
