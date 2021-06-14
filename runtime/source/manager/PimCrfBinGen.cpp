/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "manager/PimCrfBinGen.h"
#include <cmath>

namespace pim
{
namespace runtime
{
namespace manager
{
PimCrfBinGen::PimCrfBinGen() : is_gemv_tree_(true) {}

void PimCrfBinGen::gen_binary_with_loop(PimOpType op_type, int lc, uint8_t* bin_buf, int* crf_sz)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    create_pim_cmd(op_type, lc);
    change_to_binary(bin_buf, crf_sz);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

void PimCrfBinGen::create_pim_cmd(PimOpType op_type, int lc)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    if (op_type == OP_ELT_ADD) {
        std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
            PimCommand(PimCmdType::ADD, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1),
            PimCommand(PimCmdType::NOP, 23),
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK),
            PimCommand(PimCmdType::ADD, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1),
            PimCommand(PimCmdType::NOP, 23)/*,
            PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_ELT_MUL) {
        std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
            PimCommand(PimCmdType::MUL, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1),
            PimCommand(PimCmdType::NOP, 23),
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK),
            PimCommand(PimCmdType::MUL, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1),
            PimCommand(PimCmdType::NOP, 23)/*,
            PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_RELU) {
        std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0, 1),
            PimCommand(PimCmdType::NOP, 15),
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1, 0, 0, 0, 1),
            PimCommand(PimCmdType::NOP, 15) /*, PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_GEMV) {
        if (is_gemv_tree_) { 
            std::vector<PimCommand> tmp_cmds{
                PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0),
                PimCommand(PimCmdType::JUMP, 7, 2),
                PimCommand(PimCmdType::NOP, 23),
                PimCommand(PimCmdType::MUL, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::EVEN_BANK, 1),
                PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::ODD_BANK, 1, 0, 0, 0),
                PimCommand(PimCmdType::JUMP, 7, 2),
                PimCommand(PimCmdType::NOP, 23),
                PimCommand(PimCmdType::MUL, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::EVEN_BANK, 1)};
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
        } else {
            int even_lc = 8 * ceil((float)lc / 2) - 1;
            int odd_lc = 8 * (lc / 2) - 1;
            std::vector<PimCommand> tmp_cmds{
                PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0),
                PimCommand(PimCmdType::JUMP, even_lc, 2),
                PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::ODD_BANK, 1, 0, 0, 0),
                PimCommand(PimCmdType::JUMP, odd_lc, 2), PimCommand(PimCmdType::NOP, 23)};
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
        }
    } else if (op_type == OP_BN) {
        std::vector<PimCommand> tmp_cmds{PimCommand(PimCmdType::MAD, PimOpdType::GRF_A, PimOpdType::EVEN_BANK,
                                                    PimOpdType::SRF_M, PimOpdType::SRF_A, 1, 0, 0, 0),
                                         PimCommand(PimCmdType::NOP, 7),
                                         PimCommand(PimCmdType::MAD, PimOpdType::GRF_A, PimOpdType::GRF_A,
                                                    PimOpdType::SRF_M, PimOpdType::SRF_A, 1, 0, 0, 1),
                                         PimCommand(PimCmdType::NOP, 7),
                                         PimCommand(PimCmdType::MAD, PimOpdType::GRF_B, PimOpdType::ODD_BANK,
                                                    PimOpdType::SRF_M, PimOpdType::SRF_A, 1, 0, 0, 0),
                                         PimCommand(PimCmdType::NOP, 7),
                                         PimCommand(PimCmdType::MAD, PimOpdType::GRF_B, PimOpdType::GRF_B,
                                                    PimOpdType::SRF_M, PimOpdType::SRF_A, 1, 0, 0, 1),
                                         PimCommand(PimCmdType::NOP, 23)
                                         /*PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    }

    if (lc != 0 && is_gemv_tree_ == true || op_type != OP_GEMV) {
        cmds_.push_back(PimCommand(PimCmdType::JUMP, lc, cmds_.size() + 1));
    }

    cmds_.push_back(PimCommand(PimCmdType::EXIT, 0));

    int nop_cnt = 8 - cmds_.size() % 8;
    for (int i = 0; i < nop_cnt; i++) cmds_.push_back(PimCommand(PimCmdType::NOP, 0));

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}
void PimCrfBinGen::change_to_binary(uint8_t* crf_binary, int* crf_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PimCommand nop_cmd(PimCmdType::NOP, 0);
    *crf_size = cmds_.size() * sizeof(uint32_t);

    for (int i = 0; i < cmds_.size(); i++) {
        uint32_t u32_data_ = cmds_[i].to_int();
        memcpy(&crf_binary[i * 4], &u32_data_, sizeof(uint32_t));
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */
