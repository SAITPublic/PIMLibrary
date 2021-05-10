/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_VALIDATION_CHECKER_H_
#define _PIM_VALIDATION_CHECKER_H_

#include <fstream>
#include <vector>
#include "PimCmd.h"

namespace crfgen_offline
{
enum class PimCmdPairType { MUL, ADD, MAC, MAD, ETC };
enum class PimOpdPairType { GRF_A, GRF_B, EVEN_BANK, ODD_BANK, SRF_M, SRF_A, ETC };

enum class PimCamType { NOP, ALL, CAM, NO_CAM };

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

class PimValidationChecker
{
   public:
    PimValidationChecker();
    ~PimValidationChecker();

    int validation_check(std::vector<PimCommand>& pim_cmd_vec);
    int check_cmd_validation(std::vector<PimCommand>& cmds);
    int check_validate_pair(PimCommand& pim_cmd);
    int check_isa_restriction(std::vector<PimCommand>& cmds);
    int check_hazard(std::vector<PimCommand>& cmds);

    int find_next_op(std::vector<PimCommand>& cmds, int cur_idx, int* next_idx, int* num_nop);
    int detect_data_hazard(std::vector<PimCommand>& cmds, int cur_idx, int next_idx, int num_nop);
    int detect_structural_hazard(std::vector<PimCommand>& cmds, int cur_idx, int next_idx, int num_nop);

    PimCmdPairType change_cmd_type(PimCmdType cmd_type) const;
    PimOpdPairType change_opd_type(PimOpdType opd_type) const;

    int get_hazard_table_idx(PimCommand& cmd);
    int is_register(PimOpdType opd_type);
    int is_read_register(PimCommand& cmd);
};

}  // namespace crfgen_offline
#endif
