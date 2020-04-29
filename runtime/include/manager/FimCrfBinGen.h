#ifndef _FIM_CRF_BIN_GEN_H_
#define _FIM_CRF_BIN_GEN_H_

#include <vector>
#include "fim_data_types.h"
#include "manager/FimCommand.h"

namespace fim
{
namespace runtime
{
namespace manager
{
class FimCrfBinGen
{
   public:
    FimCrfBinGen();
    void gen_binary(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi, uint8_t* binary_buffer,
                    int* crf_size);
    void create_fim_cmd(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi);
    void change_to_binary(uint8_t* crf_binary, int* crf_size);

   private:
    std::vector<FimCommand> cmds_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _CRF_CODE_GENERATOR_H_ */