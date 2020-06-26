#ifndef _FIM_CRF_BIN_GEN_H_
#define _FIM_CRF_BIN_GEN_H_

#include <vector>

#include "FimCmd.h"
#include "fim_data_types.h"

namespace crfgen_offline
{
class FimCrfBinGen
{
   public:
    FimCrfBinGen();
    // void gen_binary(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi, uint8_t* binary_buffer,
    //                 int* crf_size);
    // void create_fim_cmd(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi);
    // void test_operand_check();
    void convert_to_binary(std::vector<FimCommand>& fim_cmd_vec, uint32_t* crf_binary);
    void convert_to_binary(FimCommand fim_cmd, uint32_t* crf_binary);
    // int check_toggle_condition();
};

}  // namespace crfgen_offline

#endif /* _CRF_CODE_GENERATOR_H_ */