#ifndef _FIM_VALIDATION_CHECKER_H_
#define _FIM_VALIDATION_CHECKER_H_

#include <fstream>
#include <vector>
#include "FimCmd.h"

namespace crfgen_offline
{
enum class FimCmdPairType { MUL, ADD, MAC, MAD, ETC };
enum class FimOpdPairType { GRF_A, GRF_B, EVEN_BANK, ODD_BANK, SRF_M, SRF_A, ETC };

enum class FimCamType { NOP, ALL, CAM, NO_CAM };

static const int src_pair_table[6][6][4] = {
    {{0, 0, 0, 0}, {1, 1, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 1}, {0, 1, 0, 0}},
    {{1, 1, 2, 1}, {0, 0, 0, 0}, {1, 1, 2, 1}, {1, 1, 2, 1}, {0, 0, 2, 1}, {0, 1, 0, 0}},
    {{1, 1, 1, 0}, {1, 1, 2, 1}, {1, 1, 1, 3}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}},
    {{1, 1, 1, 0}, {1, 1, 2, 1}, {0, 0, 0, 0}, {1, 1, 1, 3}, {0, 0, 0, 1}, {0, 1, 0, 0}},
    {{0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
    {{0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};

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

class FimValidationChecker
{
   public:
    FimValidationChecker();
    ~FimValidationChecker();

    int validation_check(std::vector<FimCommand>& fim_cmd_vec);
    int check_cmd_validation(std::vector<FimCommand>& cmds);
    int check_validate_pair(FimCommand& fim_cmd);
    int check_isa_restriction(std::vector<FimCommand>& cmds);
    int check_hazard(std::vector<FimCommand>& cmds);

    int find_next_op(std::vector<FimCommand>& cmds, int cur_idx, int* next_idx, int* num_nop);
    int detect_data_hazard(std::vector<FimCommand>& cmds, int cur_idx, int next_idx, int num_nop);
    int detect_structural_hazard(std::vector<FimCommand>& cmds, int cur_idx, int next_idx, int num_nop);

    FimCmdPairType change_cmd_type(FimCmdType cmd_type) const;
    FimOpdPairType change_opd_type(FimOpdType opd_type) const;

    int get_hazard_table_idx(FimCommand& cmd);
    int is_register(FimOpdType opd_type);
    int is_read_register(FimCommand& cmd);
};

}  // namespace crfgen_offline
#endif