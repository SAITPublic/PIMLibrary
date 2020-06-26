#include <iostream>
#include "fim_crf_gen_api.h"

int main(void)
{
    std::cout << "crf generator" << std::endl;
    int crf_size = 0;
    uint32_t buffer[128] = {
        0,
    };

    FimCommand test_cmd(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::GRF_A, 1, 0, 0, 0);
    // FimCommand test_cmd(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1,0,0,0);
    std::cout << ConvertToBinary(test_cmd, buffer) << std::endl;

    return 0;
}

// void test_operand_check()
// {
//     FimCmdType test_cmd[4] = {FimCmdType::MUL, FimCmdType::ADD, FimCmdType::MAC, FimCmdType::MAD};
//     FimOpdType test_src[6] = {FimOpdType::GRF_A,    FimOpdType::GRF_B, FimOpdType::EVEN_BANK,
//                               FimOpdType::ODD_BANK, FimOpdType::SRF_M, FimOpdType::SRF_A};
//     FimOpdType test_src2[7] = {FimOpdType::A_OUT,    FimOpdType::GRF_A, FimOpdType::GRF_B, FimOpdType::EVEN_BANK,
//                                FimOpdType::ODD_BANK, FimOpdType::SRF_M, FimOpdType::SRF_A};

//     cmds_.clear();

//     for (int auto_idx = 0; auto_idx < 2; auto_idx++) {
//         for (int src2_idx = 0; src2_idx < 7; src2_idx++) {
//             for (int dst_idx = 0; dst_idx < 6; dst_idx++) {
//                 for (int src0_idx = 0; src0_idx < 6; src0_idx++) {
//                     for (int src1_idx = 0; src1_idx < 6; src1_idx++) {
//                         for (int cmd_idx = 0; cmd_idx < 4; cmd_idx++) {
//                             FimCommand test_fim(test_cmd[cmd_idx], test_src[dst_idx], test_src[src0_idx],
//                                                 test_src[src1_idx], test_src2[src2_idx], auto_idx);
//                             cmds_.push_back(test_fim);
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     vc.check_cmd_validation(cmds_);
// }