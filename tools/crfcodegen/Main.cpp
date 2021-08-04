/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <iostream>
#include "pim_crf_gen_api.h"

int main(void)
{
    std::cout << "crf generator" << std::endl;
    uint32_t buffer[128] = {
        0,
    };

    PimCommand test_cmd(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::GRF_A, 1, 0, 0, 0);
    // PimCommand test_cmd(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1,0,0,0);
    std::cout << ConvertToBinary(test_cmd, buffer) << std::endl;

    return 0;
}

// void test_operand_check()
// {
//     PimCmdType test_cmd[4] = {PimCmdType::MUL, PimCmdType::ADD, PimCmdType::MAC, PimCmdType::MAD};
//     PimOpdType test_src[6] = {PimOpdType::GRF_A,    PimOpdType::GRF_B, PimOpdType::EVEN_BANK,
//                               PimOpdType::ODD_BANK, PimOpdType::SRF_M, PimOpdType::SRF_A};
//     PimOpdType test_src2[7] = {PimOpdType::A_OUT,    PimOpdType::GRF_A, PimOpdType::GRF_B, PimOpdType::EVEN_BANK,
//                                PimOpdType::ODD_BANK, PimOpdType::SRF_M, PimOpdType::SRF_A};

//     cmds_.clear();

//     for (int auto_idx = 0; auto_idx < 2; auto_idx++) {
//         for (int src2_idx = 0; src2_idx < 7; src2_idx++) {
//             for (int dst_idx = 0; dst_idx < 6; dst_idx++) {
//                 for (int src0_idx = 0; src0_idx < 6; src0_idx++) {
//                     for (int src1_idx = 0; src1_idx < 6; src1_idx++) {
//                         for (int cmd_idx = 0; cmd_idx < 4; cmd_idx++) {
//                             PimCommand test_pim(test_cmd[cmd_idx], test_src[dst_idx], test_src[src0_idx],
//                                                 test_src[src1_idx], test_src2[src2_idx], auto_idx);
//                             cmds_.push_back(test_pim);
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     vc.check_cmd_validation(cmds_);
// }
