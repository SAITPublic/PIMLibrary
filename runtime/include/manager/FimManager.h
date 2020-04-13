#ifndef _FIM_MANAGER_H_
#define _FIM_MANAGER_H_

#include "fim_data_types.h"
#include "manager/FimControlManager.h"
#include "manager/FimCrfBinGen.h"
#include "manager/FimDevice.h"
#include "manager/FimMemoryManager.h"

namespace fim
{
namespace runtime
{
namespace manager
{
class FimMemoryManager;
class FimControlManager;
class FimDevice;

class FimManager
{
   public:
    virtual ~FimManager(void);

    static FimManager* get_instance(FimRuntimeType rt_type, FimPrecision precision);

    int initialize(void);
    int deinitialize(void);
    int alloc_memory(void** ptr, size_t size, FimMemType mem_type);
    int alloc_memory(FimBo* fim_bo);
    int free_memory(void* ptr, FimMemType mem_type);
    int free_memory(FimBo* fim_bo);
    int copy_memory(void* dst, void* src, size_t size, FimMemCpyType cpy_type);
    int copy_memory(FimBo* dst, FimBo* src, FimMemCpyType);
    int convert_data_layout(void* dst, void* src, size_t size, FimOpType op_type);
    int convert_data_layout(FimBo* dst, FimBo* src, FimOpType);
    int convert_data_layout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type);

    int create_crf_binary(FimOpType op_type, int input_size, int output_size);
    uint8_t* get_crf_binary();
    int get_crf_size();

   private:
    FimManager(FimRuntimeType rt_type, FimPrecision precision);

    FimDevice* fim_device_;
    FimControlManager* fim_control_manager_;
    FimMemoryManager* fim_memory_manager_;
    FimCrfBinGen* fim_crf_generator_;

    FimRuntimeType rt_type_;
    FimPrecision precision_;

    uint8_t h_binary_buffer_[128] = {
        0,
    };
    int crf_size_;
    FimBlockInfo fbi_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_MANAGER_H_ */
