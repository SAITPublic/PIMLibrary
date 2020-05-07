#include "manager/FimCrfBinGen.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimCrfBinGen::FimCrfBinGen() {}

void FimCrfBinGen::gen_binary(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi,
                              uint8_t* binary_buffer, int* crf_size)
{
    DLOG(INFO) << "[START] " <<__FUNCTION__ << " called";
    create_fim_cmd(op_type, input_size, output_size, fbi);
    change_to_binary(binary_buffer, crf_size);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

void FimCrfBinGen::create_fim_cmd(FimOpType op_type, int input_size, int output_size, FimBlockInfo* fbi)
{
    DLOG(INFO) << "[START] " <<__FUNCTION__ << " called";
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
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

void FimCrfBinGen::change_to_binary(uint8_t* crf_binary, int* crf_size)
{
    DLOG(INFO) << "[START] " <<__FUNCTION__ << " called";
    FimCommand nop_cmd(FimCmdType::NOP, 0);
    *crf_size = cmds_.size();

    for (int i = 0; i < cmds_.size(); i++) {
        uint32_t u32_data_ = cmds_[i].to_int();
        memcpy(&crf_binary[i * 4], &u32_data_, sizeof(uint32_t));
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}
} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
