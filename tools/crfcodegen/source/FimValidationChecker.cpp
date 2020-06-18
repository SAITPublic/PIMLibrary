#include "FimValidationChecker.h"
#include <cassert>

namespace crfgen_offline
{
FimValidationChecker::FimValidationChecker() {}
FimValidationChecker::~FimValidationChecker() {}

FimCmdPairType FimValidationChecker::change_cmd_type(FimCmdType cmd_type) const
{
    FimCmdPairType changed_cmd_type;
    switch (cmd_type) {
        case FimCmdType::MUL:
            changed_cmd_type = FimCmdPairType::MUL;
            break;
        case FimCmdType::ADD:
            changed_cmd_type = FimCmdPairType::ADD;
            break;
        case FimCmdType::MAC:
            changed_cmd_type = FimCmdPairType::MAC;
            break;
        case FimCmdType::MAD:
            changed_cmd_type = FimCmdPairType::MAD;
            break;
        case FimCmdType::EXIT:
        case FimCmdType::NOP:
        case FimCmdType::JUMP:
        case FimCmdType::FILL:
        case FimCmdType::MOV:
            changed_cmd_type = FimCmdPairType::ETC;
            break;

        default:
            break;
    }
    return changed_cmd_type;
}

FimOpdPairType FimValidationChecker::change_opd_type(FimOpdType opd_type) const
{
    FimOpdPairType changed_opd_type;
    switch (opd_type) {
        case FimOpdType::EVEN_BANK:
            changed_opd_type = FimOpdPairType::EVEN_BANK;
            break;
        case FimOpdType::ODD_BANK:
            changed_opd_type = FimOpdPairType::ODD_BANK;
            break;
        case FimOpdType::GRF_A:
            changed_opd_type = FimOpdPairType::GRF_A;
            break;
        case FimOpdType::GRF_B:
            changed_opd_type = FimOpdPairType::GRF_B;
            break;
        case FimOpdType::SRF_M:
            changed_opd_type = FimOpdPairType::SRF_M;
            break;
        case FimOpdType::SRF_A:
            changed_opd_type = FimOpdPairType::SRF_A;
            break;
        case FimOpdType::A_OUT:
        case FimOpdType::M_OUT:
            changed_opd_type = FimOpdPairType::ETC;
            break;
    }

    return changed_opd_type;
}

int FimValidationChecker::check_cmd_validation(std::vector<FimCommand>& cmds)
{
    int ret = 1;

    for (int i = 0; i < cmds.size(); i++) {
        ret = check_validate_pair(cmds[i]);
        if(0 == ret)
            return ret;
    }

    return ret;
}

int FimValidationChecker::check_validate_pair(FimCommand& fim_cmd)
{
    int vaild_flag = 1;

    FimCmdPairType cmd_type = change_cmd_type(fim_cmd.type_);
    int i_cmd = static_cast<int>(cmd_type);

    if (cmd_type != FimCmdPairType::ETC) {
        FimOpdPairType src0 = change_opd_type(fim_cmd.src0_);
        FimOpdPairType src1 = change_opd_type(fim_cmd.src1_);
        FimOpdPairType src2 = change_opd_type(fim_cmd.src2_);
        int i_src0 = static_cast<int>(src0);
        int i_src1 = static_cast<int>(src1);
        int i_src2 = static_cast<int>(src2);

        if (src_pair_table[i_src0][i_src1][i_cmd] == static_cast<int>(FimCamType::NOP)) {
            std::cout << "Invalid in ISA 1.0  ( " << fim_cmd.to_str()
                      << " ) - This operation not support the current src pair." << std::endl;
            vaild_flag = 0;
        }

        if (src_pair_table[i_src0][i_src1][i_cmd] == static_cast<int>(FimCamType::CAM)) {
            if (!fim_cmd.is_auto_) {
                std::cout << "Invalid in ISA 1.0  ( " << fim_cmd.to_str()
                          << " ) - This operation and src pair support only CAM mode" << std::endl;
                vaild_flag = 0;
            }
        }

        if (src_pair_table[i_src0][i_src1][i_cmd] == static_cast<int>(FimCamType::NO_CAM)) {
            if (fim_cmd.is_auto_) {
                std::cout << "Invalid in ISA 1.0  ( " << fim_cmd.to_str()
                          << " ) - This operation and src pair not support CAM mode" << std::endl;
                vaild_flag = 0;
            }
        }

        if ((fim_cmd.src2_ != FimOpdType::SRF_A) && (fim_cmd.src2_ != FimOpdType::A_OUT)) {
            std::cout << "Invalid in ISA 1.0  ( " << fim_cmd.to_str() << " ) - The src2 use only SRF_A " << std::endl;
            vaild_flag = 0;
        }

        if ((fim_cmd.src2_ == FimOpdType::SRF_A) && (fim_cmd.type_ != FimCmdType::MAD)) {
            std::cout << "Invalid in ISA 1.0  ( " << fim_cmd.to_str() << " ) - src2 is used only in MAD operation"
                      << std::endl;
            vaild_flag = 0;
        }

        if ((fim_cmd.type_ == FimCmdType::MAC) && (fim_cmd.dst_ == FimOpdType::GRF_A)) {
            std::cout << "Invalid in ISA 1.0  ( " << fim_cmd.to_str() << " ) - MAC operation not use grf_a as dst"
                      << std::endl;
            vaild_flag = 0;
        }

        if (fim_cmd.dst_ != FimOpdType::GRF_A && fim_cmd.dst_ != FimOpdType::GRF_B) {
            std::cout << "Invalid in ISA 1.0  ( " << fim_cmd.to_str()
                      << " ) - All operation use only grf_a or grf_b as dst" << std::endl;
            vaild_flag = 0;
        }
    }

    return vaild_flag;
}

int FimValidationChecker::check_isa_restriction(std::vector<FimCommand>& cmds)
{
    int cmd_last_idx = cmds.size() - 1;
    int is_valid = 1;

    if (cmds[0].type_ == FimCmdType::NOP) {
        std::cerr << "ISA Validation error :  NOP cannot be the first command." << std::endl;
        is_valid = 0;
    }

    for (int i = 0; i < cmds.size(); i++) {
        if (cmds[i].type_ == FimCmdType::JUMP) {
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
            if (cmds[i - cmds[i].loop_offset_ + 1].type_ == FimCmdType::NOP ||
                cmds[i - cmds[i].loop_offset_ + 1].type_ == FimCmdType::JUMP) {
                std::cerr << "ISA Validation error :  JUMP target  cannot be  JUMP or NOP." << std::endl;
                is_valid = 0;
            }

            if (i != cmd_last_idx && cmds[i + 1].type_ == FimCmdType::NOP) {
                std::cerr << "ISA Validation error :  NOP cannot come immediately after JUMP." << std::endl;
                is_valid = 0;
            }
        }

        if (cmds[i].type_ == FimCmdType::NOP) {
            if (i != cmd_last_idx && cmds[i].loop_counter_ > 0 && cmds[i + 1].type_ == FimCmdType::JUMP) {
                std::cerr << "ISA Validation error :  JUMP cannot come immediately after multicycle NOP." << std::endl;
                is_valid = 0;
            }
            if (i != cmd_last_idx && cmds[i + 1].type_ == FimCmdType::NOP && cmds[i].loop_counter_ > 0 &&
                cmds[1].loop_counter_ > 0) {
                std::cerr << "ISA Validation error :  Two consecutive multicycle NOP is not supported." << std::endl;
                is_valid = 0;
            }
        }
    }

    return is_valid;
}

int FimValidationChecker::find_next_op(std::vector<FimCommand>& cmds, int cur_idx, int* next_idx, int* num_nop)
{
    for (int i = cur_idx + 1; i < cmds.size(); i++) {
        int op_idx = get_hazard_table_idx(cmds[i]);

        if (cmds[i].type_ == FimCmdType::NOP) {
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

int FimValidationChecker::check_hazard(std::vector<FimCommand>& cmds)
{
    int is_structural_hazard = 0;
    int is_data_hazard = 0;
    int cur_op = 0;
    int max_nop = 4;
    int next_idx = 0;
    int num_nop = 0;
    int num_hazard = 0;

    for (int i = 0; i < cmds.size(); i++) {
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

int FimValidationChecker::detect_structural_hazard(std::vector<FimCommand>& cmds, int cur_idx, int next_idx,
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

int FimValidationChecker::detect_data_hazard(std::vector<FimCommand>& cmds, int cur_idx, int next_idx, int num_nop)
{
    int opcode_idx = get_hazard_table_idx(cmds[cur_idx]);
    assert(opcode_idx >= 0);

    int is_read_reg = is_read_register(cmds[cur_idx]);
    int num_required_nop = data_hazard_table[is_read_reg][opcode_idx];

    int max_idx = ((int64_t)cur_idx + num_required_nop + 1) < cmds.size() ? ((int64_t)cur_idx + num_required_nop + 1) : (cmds.size());
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

int FimValidationChecker::is_register(FimOpdType opd_type)
{
    switch (opd_type) {
        case FimOpdType::GRF_A:
        case FimOpdType::GRF_B:
        case FimOpdType::SRF_M:
        case FimOpdType::SRF_A:
            return 1;
        default:
            return 0;
    }

    return 0;
}

int FimValidationChecker::is_read_register(FimCommand& cmd)
{
    if (is_register(cmd.src0_) || is_register(cmd.src1_) || cmd.type_ == FimCmdType::MAC || is_register(cmd.src2_))
        return 1;

    return 0;
}

int FimValidationChecker::get_hazard_table_idx(FimCommand& cmd)
{
    switch (cmd.type_) {
        case FimCmdType::ADD:
            return 0;
        case FimCmdType::MUL:
            return 1;
        case FimCmdType::MAC:
            return 2;
        case FimCmdType::MAD:
            return 3;
        case FimCmdType::FILL:
        case FimCmdType::MOV:
            return 4;
        case FimCmdType::EXIT:
        case FimCmdType::NOP:
        case FimCmdType::JUMP:
            return -1;

        default:
            return -1;
    }
}

}  // namespace crfgen_offline
