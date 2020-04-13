#include "manager/FimCrfBinGen.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimCrfBinGen::FimCrfBinGen()
{
    src_pair_log_.open("src_pair.txt", std::ios_base::out | std::ios_base::trunc);
    data_hazard_log_.open("data_hazard.txt", std::ios_base::out | std::ios_base::trunc);
    structural_hazard_log_.open("struct_hazard.txt", std::ios_base::out | std::ios_base::trunc);
}

void FimCrfBinGen::gen_binary(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi,
                              uint8_t* binary_buffer, int* crf_size)
{
    // std::cout <<" input_size  : " << input_size << std::endl;
    // std::cout <<" output_size : " << output_size << std::endl;
    create_fim_cmd(op_type, input_size, output_size, fbi);
    validation_check();
    change_to_binary(binary_buffer, crf_size);
}

void FimCrfBinGen::create_fim_cmd(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi)
{
    int num_transaction = (input_size / 16) / sizeof(uint16_t);
    int num_parallelism = fbi->num_fim_blocks * fbi->num_fim_chan * fbi->num_fim_rank * fbi->num_grf;
    int num_tile = num_transaction / num_parallelism;
    int num_jump_to_be_taken = 0;
    int num_jump_of_even_bank = 0;
    int num_jump_of_odd_bank = 0;

    if (op_type == OP_RELU) {
        num_jump_to_be_taken = num_tile / 2 - 1;
    } else if (op_type == OP_GEMV) {
        num_tile = ceil((double)num_transaction / (double)fbi->num_grf);
        num_jump_of_even_bank = fbi->num_grf * ceil((double)num_tile / 2) - 1;
        num_jump_of_odd_bank = fbi->num_grf * floor(num_tile / 2) - 1;
    } else {
        num_jump_to_be_taken = num_tile - 1;
    }

    if (op_type == OP_ELT_ADD) {
        std::vector<FimCommand> tmp_cmds{
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK),
            FimCommand(FimCmdType::ADD, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1),
            FimCommand(FimCmdType::NOP, 7), FimCommand(FimCmdType::JUMP, num_jump_to_be_taken, 4),
            FimCommand(FimCmdType::EXIT, 0)};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_ELT_MUL) {
        std::vector<FimCommand> tmp_cmds{
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK),
            FimCommand(FimCmdType::MUL, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1),
            FimCommand(FimCmdType::NOP, 7), FimCommand(FimCmdType::JUMP, num_jump_to_be_taken, 4),
            FimCommand(FimCmdType::EXIT, 0)};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_RELU) {
        std::vector<FimCommand> tmp_cmds{
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1, 0, 0, 0, 1),
            FimCommand(FimCmdType::NOP, 7),
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_B, FimOpdType::ODD_BANK, 1, 0, 0, 0, 1),
            FimCommand(FimCmdType::NOP, 7),
            FimCommand(FimCmdType::JUMP, num_jump_to_be_taken, 5),
            FimCommand(FimCmdType::EXIT, 0)};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_GEMV) {
        std::vector<FimCommand> tmp_cmds{
            FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1, 0, 0, 0),
            FimCommand(FimCmdType::JUMP, num_jump_of_even_bank, 2),
            FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1, 0, 0, 0),
            FimCommand(FimCmdType::JUMP, num_jump_of_odd_bank, 2),
            FimCommand(FimCmdType::NOP, 7),
            FimCommand(FimCmdType::EXIT, 0)};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    }
}

void FimCrfBinGen::change_to_binary(uint8_t* crf_binary, int* crf_size)
{
    FimCommand nop_cmd(FimCmdType::NOP, 0);
    *crf_size = cmds_.size();

    for (int i = 0; i < cmds_.size(); i++) {
        uint32_t u32_data_ = cmds_[i].to_int();
        memcpy(&crf_binary[i * 4], &u32_data_, sizeof(uint32_t));
    }
}

void FimCrfBinGen::validation_check()
{
    operand_check();
    hazard_check();
}

int FimCrfBinGen::find_next_op(int cur_idx, int* next_idx, int* num_nop)
{
    for (int i = cur_idx + 1; i < cmds_.size(); i++) {
        int op_idx = cmds_[i].get_opcode_idx();

        if (cmds_[i].type_ == FimCmdType::NOP) {
            *num_nop += cmds_[i].loop_counter_ + 1;
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

int FimCrfBinGen::hazard_check()
{
    int is_structural_hazard = 0;
    int is_data_hazard = 0;
    int cur_op = 0;
    int max_nop = 4;
    int next_idx = 0;
    int num_nop = 0;
    int num_hazard = 0;

    for (int i = 0; i < cmds_.size(); i++) {
        num_nop = 0;
        next_idx = 0;
        cur_op = cmds_[i].get_opcode_idx();
        if (cur_op == -1) {
            continue;
        }

        find_next_op(i, &next_idx, &num_nop);
        if (next_idx == 0) break;

        if (num_nop >= max_nop) {
            i = next_idx - 1;
            continue;
        }

        is_structural_hazard = detect_structural_hazard(i, next_idx, num_nop);
        is_data_hazard = detect_data_hazard(i, next_idx, num_nop);
        if (is_structural_hazard) {
            std::cerr << " error : This cmd combination generates structural hazard" << std::endl;
            std::cerr << " ISA index : " << i << ", ISA INFO ( " << cmds_[i].to_str() << " ) " << std::endl;
            num_hazard++;
        }

        if (is_data_hazard) {
            std::cerr << " error : This cmd combination generates data hazard" << std::endl;
            std::cerr << " ISA index : " << i << ", ISA INFO ( " << cmds_[i].to_str() << " ) " << std::endl;
            num_hazard++;
        }
    }

    return num_hazard;
}

int FimCrfBinGen::detect_structural_hazard(int cur_idx, int next_idx, int num_nop)
{
    int cur_op_idx = cmds_[cur_idx].get_opcode_idx();
    int next_op_idx = cmds_[next_idx].get_opcode_idx();
    int is_consecutive = cmds_[cur_idx].dst_ == cmds_[next_idx].dst_;
    int r_nonr_idx = !cmds_[cur_idx].is_read_register() * 2 + !cmds_[next_idx].is_read_register();
    int num_required_nop = structual_hazard_table[is_consecutive][r_nonr_idx][next_op_idx][cur_op_idx];

    if (num_required_nop == -1) {
        return -1;
    }

    if (num_nop < num_required_nop) {
        return 1;
    }

    return 0;
}

int FimCrfBinGen::detect_data_hazard(int cur_idx, int next_idx, int num_nop)
{
    int opcode_idx = cmds_[cur_idx].get_opcode_idx();
    int is_read_reg = cmds_[cur_idx].is_read_register();
    int num_required_nop = data_hazard_table[is_read_reg][opcode_idx];

    int max_idx = (cur_idx + num_required_nop + 1) < cmds_.size() ? (cur_idx + num_required_nop + 1) : (cmds_.size());
    int is_hazard = 0;

    // is_auto -> dst==src data hazard
    //        -> dst!=src no                  -> num nop 추가
    // not_auto -> is auto -> dst ==src   -> data hazard
    //                     -> dst != src -> no           num _nop 추가
    //          -> not auto -> dst == src and dst_idx==src_idx hazard
    //          -            -> dst != src -> num_nop만 추가 .

    // NOP,ADD, MUL, MAD MAC, JUMP, EXIT, FILL, MOVE
    for (int i = next_idx; i < max_idx; i++) {
        if (num_nop >= num_required_nop) {
            //   std::cout <<" break - num_nop : " << num_nop << "  num_required_nop : " <<num_required_nop <<std::endl;
            break;
        }

        if (cmds_[cur_idx].is_auto_) {
            if (cmds_[cur_idx].dst_ == cmds_[i].src0_ || cmds_[cur_idx].dst_ == cmds_[i].src1_) {
                is_hazard = 1;
            } else {
                num_nop += 1 + cmds_[i].is_auto_ * 7;
            }
        } else {
            if (cmds_[i].is_auto_) {
                if (cmds_[cur_idx].dst_ == cmds_[i].src0_ || cmds_[cur_idx].dst_ == cmds_[i].src1_) {
                    is_hazard = 1;
                } else {
                    num_nop += 1 + cmds_[i].is_auto_ * 7;
                }
            } else {
                if (((cmds_[cur_idx].dst_ == cmds_[i].src0_) && (cmds_[cur_idx].dst_idx_ == cmds_[i].src0_idx_)) ||
                    ((cmds_[cur_idx].dst_ == cmds_[i].src1_) && (cmds_[cur_idx].dst_idx_ == cmds_[i].src1_idx_))) {
                    is_hazard = 1;
                } else {
                    num_nop += 1 + cmds_[i].is_auto_ * 7;
                }
            }
        }
    }

    return is_hazard;
}

void FimCrfBinGen::operand_check()
{
    int a = 0;
    for (int i = 0; i < cmds_.size(); i++) {
        if (i == 0) src_pair_log_ << " ------------------- NO CAM mode -------------------" << std::endl;

        if (i == 144 * 7 * 6)
            src_pair_log_ << std::endl
                          << std::endl
                          << " --------------------  CAM mode --------------------" << std::endl;
        // if (i == 144)
        //     src_pair_log_ <<std::endl<< std::endl<< std::endl << " operand check - NOCAM mode" << std::endl;

        if (i % 144 == 0) {
            src_pair_log_ << std::endl << std::endl << "(" << a++ << ")";
            src_pair_log_ << " dst : " << cmds_[i].opd_to_str(cmds_[i].dst_)
                          << " src2 : " << cmds_[i].opd_to_str(cmds_[i].src2_);
        }

        if (i % 24 == 0) src_pair_log_ << std::endl;
        if (i % 4 == 0) src_pair_log_ << " ";

        if (!cmds_[i].validation_check(src_pair_log_)) {
            // std::cerr << " error : src or dst is not valid." << std::endl;
            // std::cerr << " ISA index : " << i << ", ISA INFO ( " << cmds_[i].to_str() << " ) " << std::endl;
        }
    }
    src_pair_log_ << std::endl;
}

void FimCrfBinGen::test_structural_hazard_check()
{
    FimCmdType test_cmd[6] = {FimCmdType::ADD, FimCmdType::MUL, FimCmdType::MAC,
                              FimCmdType::MAD, FimCmdType::MOV, FimCmdType::FILL};

    int num_required_nop;
    FimOpdType cur_dst, next_dst;
    FimOpdType cur_src0, cur_src1, cur_src2;
    FimOpdType next_src0, next_src1, next_src2;

    for (int consecutive_idx = 0; consecutive_idx < 2; consecutive_idx++) {
        for (int r_nonr_idx = 0; r_nonr_idx < 4; r_nonr_idx++) {
            for (int next_idx = 0; next_idx < 6; next_idx++) {
                for (int cur_idx = 0; cur_idx < 6; cur_idx++) {
                    cmds_.clear();
                    num_required_nop = structual_hazard_table[consecutive_idx][r_nonr_idx][next_idx][cur_idx];

                    if (consecutive_idx == 0) {
                        cur_dst = FimOpdType::GRF_B;
                        next_dst = FimOpdType::GRF_A;
                    } else {
                        cur_dst = FimOpdType::GRF_B;
                        next_dst = FimOpdType::GRF_B;
                    }

                    if (r_nonr_idx == 0) {
                        cur_src0 = next_src0 = FimOpdType::GRF_A;
                        cur_src1 = next_src1 = FimOpdType::EVEN_BANK;
                    } else if (r_nonr_idx == 1) {
                        cur_src0 = FimOpdType::GRF_A;
                        cur_src1 = FimOpdType::EVEN_BANK;
                        next_src0 = FimOpdType::EVEN_BANK;
                        next_src1 = FimOpdType::EVEN_BANK;
                    } else if (r_nonr_idx == 2) {
                        cur_src0 = FimOpdType::EVEN_BANK;
                        cur_src1 = FimOpdType::EVEN_BANK;
                        next_src0 = FimOpdType::GRF_A;
                        next_src1 = FimOpdType::EVEN_BANK;
                    } else if (r_nonr_idx == 3) {
                        cur_src0 = FimOpdType::EVEN_BANK;
                        cur_src1 = FimOpdType::EVEN_BANK;
                        next_src0 = FimOpdType::EVEN_BANK;
                        next_src1 = FimOpdType::EVEN_BANK;
                    }

                    if (test_cmd[cur_idx] == FimCmdType::MAD) {
                        cur_src2 = FimOpdType::SRF_A;
                    } else {
                        cur_src2 = FimOpdType::A_OUT;
                    }

                    if (test_cmd[next_idx] == FimCmdType::MAD) {
                        next_src2 = FimOpdType::SRF_A;
                    } else {
                        next_src2 = FimOpdType::A_OUT;
                    }

                    FimCommand cur_cmd;
                    FimCommand next_cmd;
                    if (test_cmd[cur_idx] == FimCmdType::FILL || test_cmd[cur_idx] == FimCmdType::MOV) {
                        FimCommand tcur_cmd(test_cmd[cur_idx], cur_dst, cur_src0);
                        cur_cmd = tcur_cmd;
                    } else {
                        FimCommand tcur_cmd(test_cmd[cur_idx], cur_dst, cur_src0, cur_src1, cur_src2, 1);
                        cur_cmd = tcur_cmd;
                    }

                    num_required_nop = 4;
                    // if ( num_required_nop< 0 ) num_required_nop = 0;
                    FimCommand nop(FimCmdType::NOP, num_required_nop - 1);

                    if (test_cmd[next_idx] == FimCmdType::FILL || test_cmd[next_idx] == FimCmdType::MOV) {
                        FimCommand tnext_cmd(test_cmd[next_idx], next_dst, next_src0);
                        next_cmd = tnext_cmd;

                    } else {
                        FimCommand tnext_cmd(test_cmd[next_idx], next_dst, next_src0, next_src1, next_src2, 1);
                        next_cmd = tnext_cmd;
                    }

                    if (r_nonr_idx != (!cur_cmd.is_read_register() * 2 + !next_cmd.is_read_register())) continue;

                    // cout << "r_nor_idx :" << !cur_cmd.is_read_register() << " " <<  !next_cmd.is_read_register()
                    // <<endl;

                    cmds_.push_back(cur_cmd);
                    cmds_.push_back(nop);
                    cmds_.push_back(next_cmd);
                    // if ( num_required_nop > 0){
                    // }

                    std::cout << " cur_cmd : " << cur_cmd.to_str() << " next_cmd : " << next_cmd.to_str()
                              << "r_nop : " << num_required_nop << std::endl;
                    validation_check();
                    std::cout << std::endl;
                }
            }
        }
    }
}

void FimCrfBinGen::test_data_hazard_check()
{
    // std::vector<FimCommand> tmp_cmds{
    //         FimCommand(FimCmdType::ADD, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1),
    //         FimCommand(FimCmdType::NOP, 1),
    //         FimCommand(FimCmdType::MUL, FimOpdType::GRF_A, FimOpdType::GRF_B, FimOpdType::ODD_BANK, 1),
    //         FimCommand(FimCmdType::NOP, 1),
    //         FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1),
    //         FimCommand(FimCmdType::NOP, 2),
    //         FimCommand(FimCmdType::MAD, FimOpdType::GRF_B, FimOpdType::GRF_B, FimOpdType::SRF_M, FimOpdType::SRF_A,
    //         1), FimCommand(FimCmdType::NOP, 2), FimCommand(FimCmdType::MOV, FimOpdType::GRF_B, FimOpdType::GRF_B),
    //         FimCommand(FimCmdType::NOP, 0),
    //         FimCommand(FimCmdType::FILL, FimOpdType::GRF_B, FimOpdType::GRF_B),
    //         //FimCommand(FimCmdType::ADD, FimOpdType::GRF_A, FimOpdType::ODD_BANK, FimOpdType::ODD_BANK, 1),
    //         //FimCommand(FimCmdType::JUMP, num_jump_to_be_taken, 4),
    //         FimCommand(FimCmdType::EXIT, 0)};

    std::vector<FimCommand> tmp_cmds{
        // FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::ODD_BANK, FimOpdType::ODD_BANK, 1),
        // FimCommand(FimCmdType::NOP, 2),
        // FimCommand(FimCmdType::MUL, FimOpdType::GRF_A, FimOpdType::GRF_B, FimOpdType::ODD_BANK, 1),
        // FimCommand(FimCmdType::NOP, 1),
        // FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1),
        // FimCommand(FimCmdType::NOP, 2),
        // FimCommand(FimCmdType::MAD, FimOpdType::GRF_B, FimOpdType::GRF_B, FimOpdType::SRF_M, FimOpdType::SRF_A, 1),
        // FimCommand(FimCmdType::NOP, 2),
        FimCommand(FimCmdType::MOV, FimOpdType::GRF_B, FimOpdType::ODD_BANK), FimCommand(FimCmdType::NOP, 0),
        FimCommand(FimCmdType::FILL, FimOpdType::GRF_B, FimOpdType::GRF_B),
        // FimCommand(FimCmdType::ADD, FimOpdType::GRF_A, FimOpdType::ODD_BANK, FimOpdType::ODD_BANK, 1),
        // FimCommand(FimCmdType::JUMP, num_jump_to_be_taken, 4),
        FimCommand(FimCmdType::EXIT, 0)};

    cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());

    validation_check();
}

void FimCrfBinGen::test_operand_check()
{
    FimCmdType test_cmd[4] = {FimCmdType::MUL, FimCmdType::ADD, FimCmdType::MAC, FimCmdType::MAD};
    FimOpdType test_src[6] = {FimOpdType::GRF_A,    FimOpdType::GRF_B, FimOpdType::EVEN_BANK,
                              FimOpdType::ODD_BANK, FimOpdType::SRF_M, FimOpdType::SRF_A};
    FimOpdType test_src2[7] = {FimOpdType::A_OUT,    FimOpdType::GRF_A, FimOpdType::GRF_B, FimOpdType::EVEN_BANK,
                               FimOpdType::ODD_BANK, FimOpdType::SRF_M, FimOpdType::SRF_A};

    cmds_.clear();

    for (int auto_idx = 0; auto_idx < 2; auto_idx++) {
        for (int src2_idx = 0; src2_idx < 7; src2_idx++) {
            for (int dst_idx = 0; dst_idx < 6; dst_idx++) {
                for (int src0_idx = 0; src0_idx < 6; src0_idx++) {
                    for (int src1_idx = 0; src1_idx < 6; src1_idx++) {
                        for (int cmd_idx = 0; cmd_idx < 4; cmd_idx++) {
                            FimCommand test_fim(test_cmd[cmd_idx], test_src[dst_idx], test_src[src0_idx],
                                                test_src[src1_idx], test_src2[src2_idx], auto_idx);
                            cmds_.push_back(test_fim);
                        }
                    }
                }
            }
        }
    }

    operand_check();
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
