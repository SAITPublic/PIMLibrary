/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "PimValidationChecker.h"
#include <cassert>

namespace crfgen_offline
{
PimValidationChecker::PimValidationChecker() {}
PimValidationChecker::~PimValidationChecker() {}

PimCmdPairType PimValidationChecker::change_cmd_type(PimCmdType cmd_type) const
{
    PimCmdPairType changed_cmd_type{};
    switch (cmd_type) {
        case PimCmdType::MUL:
            changed_cmd_type = PimCmdPairType::MUL;
            break;
        case PimCmdType::ADD:
            changed_cmd_type = PimCmdPairType::ADD;
            break;
        case PimCmdType::MAC:
            changed_cmd_type = PimCmdPairType::MAC;
            break;
        case PimCmdType::MAD:
            changed_cmd_type = PimCmdPairType::MAD;
            break;
        case PimCmdType::EXIT:
        case PimCmdType::NOP:
        case PimCmdType::JUMP:
        case PimCmdType::FILL:
        case PimCmdType::MOV:
            changed_cmd_type = PimCmdPairType::ETC;
            break;

        default:
            break;
    }
    return changed_cmd_type;
}

PimOpdPairType PimValidationChecker::change_opd_type(PimOpdType opd_type) const
{
    PimOpdPairType changed_opd_type{};  // made a change here. added {} to initialize the enum class.
    switch (opd_type) {
        case PimOpdType::EVEN_BANK:
            changed_opd_type = PimOpdPairType::EVEN_BANK;
            break;
        case PimOpdType::ODD_BANK:
            changed_opd_type = PimOpdPairType::ODD_BANK;
            break;
        case PimOpdType::GRF_A:
            changed_opd_type = PimOpdPairType::GRF_A;
            break;
        case PimOpdType::GRF_B:
            changed_opd_type = PimOpdPairType::GRF_B;
            break;
        case PimOpdType::SRF_M:
            changed_opd_type = PimOpdPairType::SRF_M;
            break;
        case PimOpdType::SRF_A:
            changed_opd_type = PimOpdPairType::SRF_A;
            break;
        case PimOpdType::A_OUT:
        case PimOpdType::M_OUT:
            changed_opd_type = PimOpdPairType::ETC;
            break;
    }

    return changed_opd_type;
}

int PimValidationChecker::check_cmd_validation(std::vector<PimCommand>& cmds)
{
    int ret = 1;

    for (int i = 0; i < (int)cmds.size(); i++) {
        ret = check_validate_pair(cmds[i]);
        if (0 == ret) return ret;
    }

    return ret;
}

int PimValidationChecker::check_validate_pair(PimCommand& pim_cmd)
{
    int vaild_flag = 1;

    PimCmdPairType cmd_type = change_cmd_type(pim_cmd.type_);
    int i_cmd = static_cast<int>(cmd_type);

    if (cmd_type != PimCmdPairType::ETC) {
        PimOpdPairType src0 = change_opd_type(pim_cmd.src0_);
        PimOpdPairType src1 = change_opd_type(pim_cmd.src1_);
        // PimOpdPairType src2 = change_opd_type(pim_cmd.src2_);
        int i_src0 = static_cast<int>(src0);
        int i_src1 = static_cast<int>(src1);
        // int i_src2 = static_cast<int>(src2);

        if (src_pair_table[i_src0][i_src1][i_cmd] == static_cast<int>(PimCamType::NOP)) {
            std::cout << "Invalid in ISA 1.0  ( " << pim_cmd.to_str()
                      << " ) - This operation not support the current src pair." << std::endl;
            vaild_flag = 0;
        }

        if (src_pair_table[i_src0][i_src1][i_cmd] == static_cast<int>(PimCamType::CAM)) {
            if (!pim_cmd.is_auto_) {
                std::cout << "Invalid in ISA 1.0  ( " << pim_cmd.to_str()
                          << " ) - This operation and src pair support only CAM mode" << std::endl;
                vaild_flag = 0;
            }
        }

        if (src_pair_table[i_src0][i_src1][i_cmd] == static_cast<int>(PimCamType::NO_CAM)) {
            if (pim_cmd.is_auto_) {
                std::cout << "Invalid in ISA 1.0  ( " << pim_cmd.to_str()
                          << " ) - This operation and src pair not support CAM mode" << std::endl;
                vaild_flag = 0;
            }
        }

        if ((pim_cmd.src2_ != PimOpdType::SRF_A) && (pim_cmd.src2_ != PimOpdType::A_OUT)) {
            std::cout << "Invalid in ISA 1.0  ( " << pim_cmd.to_str() << " ) - The src2 use only SRF_A " << std::endl;
            vaild_flag = 0;
        }

        if ((pim_cmd.src2_ == PimOpdType::SRF_A) && (pim_cmd.type_ != PimCmdType::MAD)) {
            std::cout << "Invalid in ISA 1.0  ( " << pim_cmd.to_str() << " ) - src2 is used only in MAD operation"
                      << std::endl;
            vaild_flag = 0;
        }

        if ((pim_cmd.type_ == PimCmdType::MAC) && (pim_cmd.dst_ == PimOpdType::GRF_A)) {
            std::cout << "Invalid in ISA 1.0  ( " << pim_cmd.to_str() << " ) - MAC operation not use grf_a as dst"
                      << std::endl;
            vaild_flag = 0;
        }

        if (pim_cmd.dst_ != PimOpdType::GRF_A && pim_cmd.dst_ != PimOpdType::GRF_B) {
            std::cout << "Invalid in ISA 1.0  ( " << pim_cmd.to_str()
                      << " ) - All operation use only grf_a or grf_b as dst" << std::endl;
            vaild_flag = 0;
        }
    }

    return vaild_flag;
}

int PimValidationChecker::check_isa_restriction(std::vector<PimCommand>& cmds)
{
    int cmd_last_idx = cmds.size() - 1;
    int is_valid = 1;

    if (cmds[0].type_ == PimCmdType::NOP) {
        std::cerr << "ISA Validation error :  NOP cannot be the first command." << std::endl;
        is_valid = 0;
    }

    for (int i = 0; i < (int)cmds.size(); i++) {
        if (cmds[i].type_ == PimCmdType::JUMP) {
            if (cmds[i].loop_counter_ == 0) {
                std::cerr << "ISA Validation error :  JUMP loop index cannot be zero." << std::endl;
                is_valid = 0;
            }
            if (i - cmds[i].loop_offset_ + 1 < 0) {
                std::cout << " i : " << i << std::endl;
                std::cout << " cmds[i].loop_offset_ : " << cmds[i].loop_offset_ << std::endl;
                std::cerr << "ISA Validation error :  JUMP is out of range." << std::endl;
                is_valid = 0;
                continue;
            }
            if (cmds[i - cmds[i].loop_offset_ + 1].type_ == PimCmdType::NOP ||
                cmds[i - cmds[i].loop_offset_ + 1].type_ == PimCmdType::JUMP) {
                std::cerr << "ISA Validation error :  JUMP target  cannot be  JUMP or NOP." << std::endl;
                is_valid = 0;
            }

            if (i != cmd_last_idx && cmds[i + 1].type_ == PimCmdType::NOP) {
                std::cerr << "ISA Validation error :  NOP cannot come immediately after JUMP." << std::endl;
                is_valid = 0;
            }
        }

        if (cmds[i].type_ == PimCmdType::NOP) {
            if (i != cmd_last_idx && cmds[i].loop_counter_ > 0 && cmds[i + 1].type_ == PimCmdType::JUMP) {
                std::cerr << "ISA Validation error :  JUMP cannot come immediately after multicycle NOP." << std::endl;
                is_valid = 0;
            }
            if (i != cmd_last_idx && cmds[i + 1].type_ == PimCmdType::NOP && cmds[i].loop_counter_ > 0 &&
                cmds[1].loop_counter_ > 0) {
                std::cerr << "ISA Validation error :  Two consecutive multicycle NOP is not supported." << std::endl;
                is_valid = 0;
            }
        }
    }

    return is_valid;
}

int PimValidationChecker::find_next_op(std::vector<PimCommand>& cmds, int cur_idx, int* next_idx, int* num_nop)
{
    for (int i = cur_idx + 1; i < (int)cmds.size(); i++) {
        int op_idx = get_hazard_table_idx(cmds[i]);

        if (cmds[i].type_ == PimCmdType::NOP) {
            *num_nop += cmds[i].loop_counter_ + 1;
        }

        if (op_idx == -1) {
            continue;
        } else {
            *next_idx = i;
            break;
        }
    }

    return 1;
}

int PimValidationChecker::check_hazard(std::vector<PimCommand>& cmds)
{
    int is_structural_hazard = 0;
    int is_data_hazard = 0;
    int cur_op = 0;
    int max_nop = 4;
    int next_idx = 0;
    int num_nop = 0;
    int num_hazard = 0;

    for (int i = 0; i < (int)cmds.size(); i++) {
        num_nop = 0;
        next_idx = 0;
        cur_op = get_hazard_table_idx(cmds[i]);
        if (cur_op == -1) {
            continue;
        }

        find_next_op(cmds, i, &next_idx, &num_nop);
        if (next_idx == 0) break;

        if (num_nop >= max_nop) {
            i = next_idx - 1;
            continue;
        }

        is_structural_hazard = detect_structural_hazard(cmds, i, next_idx, num_nop);
        is_data_hazard = detect_data_hazard(cmds, i, next_idx, num_nop);
        if (is_structural_hazard) {
            std::cerr << " error : This cmd combination generates structural hazard" << std::endl;
            std::cerr << " ISA index : " << i << ", ISA INFO ( " << cmds[i].to_str() << " ) " << std::endl;
            num_hazard++;
        }

        if (is_data_hazard) {
            std::cerr << " error : This cmd combination generates data hazard" << std::endl;
            std::cerr << " ISA index : " << i << ", ISA INFO ( " << cmds[i].to_str() << " ) " << std::endl;
            num_hazard++;
        }
    }

    return num_hazard;
}

int PimValidationChecker::detect_structural_hazard(std::vector<PimCommand>& cmds, int cur_idx, int next_idx,
                                                   int num_nop)
{
    int cur_op_idx, next_op_idx;

    cur_op_idx = get_hazard_table_idx(cmds[cur_idx]);
    assert(cur_op_idx >= 0);

    next_op_idx = get_hazard_table_idx(cmds[next_idx]);
    assert(next_op_idx >= 0);

    int is_consecutive = cmds[cur_idx].dst_ == cmds[next_idx].dst_;
    int r_nonr_idx = !is_read_register(cmds[cur_idx]) * 2 + !is_read_register(cmds[next_idx]);
    int num_required_nop = structual_hazard_table[is_consecutive][r_nonr_idx][next_op_idx][cur_op_idx];

    if (num_required_nop == -1) {
        return -1;
    }

    if (num_nop < num_required_nop) {
        return 1;
    }

    return 0;
}

int PimValidationChecker::detect_data_hazard(std::vector<PimCommand>& cmds, int cur_idx, int next_idx, int num_nop)
{
    int opcode_idx = get_hazard_table_idx(cmds[cur_idx]);
    assert(opcode_idx >= 0);

    int is_read_reg = is_read_register(cmds[cur_idx]);
    int num_required_nop = data_hazard_table[is_read_reg][opcode_idx];

    int max_idx = ((int64_t)cur_idx + num_required_nop + 1) < (int64_t)cmds.size()
                      ? ((int64_t)cur_idx + num_required_nop + 1)
                      : (cmds.size());
    int is_hazard = 0;

    for (int i = next_idx; i < max_idx; i++) {
        if (num_nop >= num_required_nop) {
            break;
        }

        if (cmds[cur_idx].is_auto_) {
            if (cmds[cur_idx].dst_ == cmds[i].src0_ || cmds[cur_idx].dst_ == cmds[i].src1_) {
                is_hazard = 1;
            } else {
                num_nop += 1 + cmds[i].is_auto_ * 7;
            }
        } else {
            if (cmds[i].is_auto_) {
                if (cmds[cur_idx].dst_ == cmds[i].src0_ || cmds[cur_idx].dst_ == cmds[i].src1_) {
                    is_hazard = 1;
                } else {
                    num_nop += 1 + cmds[i].is_auto_ * 7;
                }
            } else {
                if (((cmds[cur_idx].dst_ == cmds[i].src0_) && (cmds[cur_idx].dst_idx_ == cmds[i].src0_idx_)) ||
                    ((cmds[cur_idx].dst_ == cmds[i].src1_) && (cmds[cur_idx].dst_idx_ == cmds[i].src1_idx_))) {
                    is_hazard = 1;
                } else {
                    num_nop += 1 + cmds[i].is_auto_ * 7;
                }
            }
        }
    }

    return is_hazard;
}

int PimValidationChecker::is_register(PimOpdType opd_type)
{
    switch (opd_type) {
        case PimOpdType::GRF_A:
        case PimOpdType::GRF_B:
        case PimOpdType::SRF_M:
        case PimOpdType::SRF_A:
            return 1;
        default:
            return 0;
    }

    return 0;
}

int PimValidationChecker::is_read_register(PimCommand& cmd)
{
    if (is_register(cmd.src0_) || is_register(cmd.src1_) || cmd.type_ == PimCmdType::MAC || is_register(cmd.src2_))
        return 1;

    return 0;
}

int PimValidationChecker::get_hazard_table_idx(PimCommand& cmd)
{
    switch (cmd.type_) {
        case PimCmdType::ADD:
            return 0;
        case PimCmdType::MUL:
            return 1;
        case PimCmdType::MAC:
            return 2;
        case PimCmdType::MAD:
            return 3;
        case PimCmdType::FILL:
        case PimCmdType::MOV:
            return 4;
        case PimCmdType::EXIT:
        case PimCmdType::NOP:
        case PimCmdType::JUMP:
            return -1;

        default:
            return -1;
    }
}

int PimValidationChecker::validation_check(std::vector<PimCommand>& pim_cmd_vec)
{
    int ret = 0;

    ret = check_cmd_validation(pim_cmd_vec);
    ret &= check_isa_restriction(pim_cmd_vec);
    ret &= check_hazard(pim_cmd_vec);

    return ret;
}

}  // namespace crfgen_offline
