#include "FimCrfBinGen.h"

namespace crfgen_offline
{
FimCrfBinGen::FimCrfBinGen() {}

void FimCrfBinGen::gen_binary(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi,
                              uint8_t* binary_buffer, int* crf_size)
{
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

    cmds_.clear();

    if (op_type == OP_RELU) {
        num_jump_to_be_taken = num_tile / 2 - 1;
    } else if (op_type == OP_GEMV) {
        num_tile = ceil((double)num_transaction / (double)fbi->num_grf);
        num_jump_of_even_bank = fbi->num_grf * ceil((double)num_tile / 2) - 1;
        num_jump_of_odd_bank = fbi->num_grf * floor((double)num_tile / 2) - 1;
    } else {
        num_jump_to_be_taken = num_tile - 1;
    }

    switch (op_type) {
        case OP_ELT_ADD: {
            std::vector<FimCommand> tmp_cmds{
                FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK),
                FimCommand(FimCmdType::ADD, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1),
                FimCommand(FimCmdType::NOP, 7)};
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
            break;
        }
        case OP_ELT_MUL: {
            std::vector<FimCommand> tmp_cmds{
                FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK),
                FimCommand(FimCmdType::MUL, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1),
                FimCommand(FimCmdType::NOP, 7)};
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
            break;
        }
        case OP_RELU: {
            std::vector<FimCommand> tmp_cmds{
                FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1, 0, 0, 0, 1),
                FimCommand(FimCmdType::NOP, 7),
                FimCommand(FimCmdType::FILL, FimOpdType::GRF_B, FimOpdType::ODD_BANK, 1, 0, 0, 0, 1),
                FimCommand(FimCmdType::NOP, 7)};
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
            break;
        }
        case OP_GEMV: {
            std::vector<FimCommand> tmp_cmds{
                FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1, 0, 0, 0),
                FimCommand(FimCmdType::JUMP, num_jump_of_even_bank, 2),
                FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1, 0, 0, 0),
                FimCommand(FimCmdType::JUMP, num_jump_of_odd_bank, 2), FimCommand(FimCmdType::NOP, 7)};
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
            break;
        }
        case OP_BN: {
            std::vector<FimCommand> tmp_cmds{FimCommand(FimCmdType::MAD, FimOpdType::GRF_A, FimOpdType::EVEN_BANK,
                                                        FimOpdType::SRF_M, FimOpdType::SRF_A, 1, 0, 0, 0),
                                             FimCommand(FimCmdType::MAD, FimOpdType::GRF_A, FimOpdType::GRF_A,
                                                        FimOpdType::SRF_M, FimOpdType::SRF_A, 1, 0, 0, 1),
                                             FimCommand(FimCmdType::NOP, 7),
                                             FimCommand(FimCmdType::MAD, FimOpdType::GRF_B, FimOpdType::ODD_BANK,
                                                        FimOpdType::SRF_M, FimOpdType::SRF_A, 1, 0, 0, 0),
                                             FimCommand(FimCmdType::MAD, FimOpdType::GRF_B, FimOpdType::GRF_B,
                                                        FimOpdType::SRF_M, FimOpdType::SRF_A, 1, 0, 0, 1),
                                             FimCommand(FimCmdType::NOP, 7),
                                             FimCommand(FimCmdType::NOP, 0)};
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
            break;
        }
        default:
            break;
    }

    if (num_jump_to_be_taken != 0) {
        cmds_.push_back(FimCommand(FimCmdType::JUMP, num_jump_to_be_taken, cmds_.size() + 1));
    }

    cmds_.push_back(FimCommand(FimCmdType::EXIT, 0));
}

void FimCrfBinGen::change_to_binary(uint8_t* crf_binary, int* crf_size)
{
    *crf_size = cmds_.size();

    for (int i = 0; i < cmds_.size(); i++) {
        uint32_t u32_data_ = cmds_[i].to_int();
        memcpy(&crf_binary[i * 4], &u32_data_, sizeof(uint32_t));
    }
}

void FimCrfBinGen::validation_check()
{
    vc_.check_cmd_validation(cmds_);
    vc_.check_isa_restriction(cmds_);
    vc_.check_hazard(cmds_);
}

int FimCrfBinGen::check_toggle_condition()
{
    int use_even = 0;
    int use_odd = 0;

    for (int i = 0; i < cmds_.size(); i++) {
        if (cmds_[i].src0_ == FimOpdType::EVEN_BANK || cmds_[i].src1_ == FimOpdType::EVEN_BANK) {
            use_even = 1;
        }
        if (cmds_[i].src0_ == FimOpdType::ODD_BANK || cmds_[i].src1_ == FimOpdType::ODD_BANK) {
            use_odd = 1;
        }
    }

    if (use_even && use_odd)
        return 0;
    else if (!use_even && use_odd)
        return 1;
    else if (use_even && !use_odd)
        return 2;
    else
        return -1;
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

    vc_.check_cmd_validation(cmds_);
}

}  // namespace crfgen_offline
