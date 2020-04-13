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
static const int data_hazard_table[2][5] = {
    {1, 1, -1, -1, 0},
    {2, 2, 3, 3, 1},
};

static const int structual_hazard_table[2][4][5][5] = {

    {{{1, 1, 1, 2, 0}, {1, 1, 1, 2, 0}, {0, 0, 0, 1, 0}, {0, 0, 1, 1, 0}, {3, 3, 3, 4, 1}},
     {{2, 2, 2, 3, 0}, {2, 2, 2, 3, 0}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {3, 3, 3, 4, 1}},
     {{0, 0, -1, -1, 0}, {0, 0, -1, -1, 0}, {0, 0, -1, -1, 0}, {0, 0, -1, -1, 0}, {2, 2, -1, -1, 1}},
     {{1, 1, -1, -1, 0}, {1, 1, -1, -1, 0}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {2, 2, -1, -1, 1}}},

    {{{0, 0, 1, 1, 0}, {0, 0, 1, 1, 0}, {0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {2, 2, 3, 3, 0}},
     {{1, 1, 2, 2, 0}, {1, 1, 2, 2, 0}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {2, 2, 3, 3, 0}},
     {{0, 0, -1, -1, 0}, {0, 0, -1, -1, 0}, {0, 0, -1, -1, 0}, {0, 0, -1, -1, 0}, {1, 1, -1, -1, 0}},
     {{0, 0, -1, -1, 0}, {0, 0, -1, -1, 0}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {1, 1, -1, -1, 0}}}

};

class FimCrfBinGen
{
   public:
    FimCrfBinGen();
    void gen_binary(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi, uint8_t* binary_buffer,
                    int* crf_size);
    void create_fim_cmd(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi);
    void change_to_binary(uint8_t* crf_binary, int* crf_size);

    void validation_check();
    int hazard_check();
    void operand_check();

    int find_next_op(int cur_idx, int* next_idx, int* num_nop);
    int detect_data_hazard(int cur_idx, int next_idx, int num_nop);
    int detect_structural_hazard(int cur_idx, int next_idx, int num_nop);

    void test_operand_check();
    void test_data_hazard_check();
    void test_structural_hazard_check();
    std::ofstream src_pair_log_;
    std::ofstream data_hazard_log_;
    std::ofstream structural_hazard_log_;

   private:
    std::vector<FimCommand> cmds_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _CRF_CODE_GENERATOR_H_ */