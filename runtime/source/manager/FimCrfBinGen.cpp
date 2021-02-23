#include "manager/FimCrfBinGen.h"
#include <cmath>

namespace fim
{
namespace runtime
{
namespace manager
{
FimCrfBinGen::FimCrfBinGen() {}

void FimCrfBinGen::gen_binary_with_loop(FimOpType op_type, int lc, uint8_t* bin_buf, int* crf_sz)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    create_fim_cmd(op_type, lc);
    change_to_binary(bin_buf, crf_sz);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

void FimCrfBinGen::create_fim_cmd(FimOpType op_type, int lc)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    if (op_type == OP_ELT_ADD) {
        std::vector<FimCommand> tmp_cmds{
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK),
            FimCommand(FimCmdType::ADD, FimOpdType::GRF_A, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1),
            FimCommand(FimCmdType::NOP, 23),
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_B, FimOpdType::ODD_BANK),
            FimCommand(FimCmdType::ADD, FimOpdType::GRF_B, FimOpdType::GRF_B, FimOpdType::ODD_BANK, 1),
            FimCommand(FimCmdType::NOP, 23)/*,
            FimCommand(FimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_ELT_MUL) {
        std::vector<FimCommand> tmp_cmds{
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK),
            FimCommand(FimCmdType::MUL, FimOpdType::GRF_A, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1),
            FimCommand(FimCmdType::NOP, 23),
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_B, FimOpdType::ODD_BANK),
            FimCommand(FimCmdType::MUL, FimOpdType::GRF_B, FimOpdType::GRF_B, FimOpdType::ODD_BANK, 1),
            FimCommand(FimCmdType::NOP, 23)/*,
            FimCommand(FimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_RELU) {
        std::vector<FimCommand> tmp_cmds{
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1, 0, 0, 0, 1),
            FimCommand(FimCmdType::NOP, 15),
            FimCommand(FimCmdType::FILL, FimOpdType::GRF_B, FimOpdType::ODD_BANK, 1, 0, 0, 0, 1),
            FimCommand(FimCmdType::NOP, 15) /*, FimCommand(FimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_GEMV) {
        int even_lc = 8 * ceil((float)lc / 2) - 1;
        int odd_lc = 8 * (lc / 2) - 1;
        std::vector<FimCommand> tmp_cmds{
            FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::EVEN_BANK, 1, 0, 0, 0),
            FimCommand(FimCmdType::JUMP, even_lc, 2),
            FimCommand(FimCmdType::MAC, FimOpdType::GRF_B, FimOpdType::GRF_A, FimOpdType::ODD_BANK, 1, 0, 0, 0),
            FimCommand(FimCmdType::JUMP, odd_lc, 2), FimCommand(FimCmdType::NOP, 23)};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_BN) {
        std::vector<FimCommand> tmp_cmds{FimCommand(FimCmdType::MAD, FimOpdType::GRF_A, FimOpdType::EVEN_BANK,
                                                    FimOpdType::SRF_M, FimOpdType::SRF_A, 1, 0, 0, 0),
                                         FimCommand(FimCmdType::NOP, 7),
                                         FimCommand(FimCmdType::MAD, FimOpdType::GRF_A, FimOpdType::GRF_A,
                                                    FimOpdType::SRF_M, FimOpdType::SRF_A, 1, 0, 0, 1),
                                         FimCommand(FimCmdType::NOP, 7),
                                         FimCommand(FimCmdType::MAD, FimOpdType::GRF_B, FimOpdType::ODD_BANK,
                                                    FimOpdType::SRF_M, FimOpdType::SRF_A, 1, 0, 0, 0),
                                         FimCommand(FimCmdType::NOP, 7),
                                         FimCommand(FimCmdType::MAD, FimOpdType::GRF_B, FimOpdType::GRF_B,
                                                    FimOpdType::SRF_M, FimOpdType::SRF_A, 1, 0, 0, 1),
                                         FimCommand(FimCmdType::NOP, 23)
                                         /*FimCommand(FimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    }

    if (lc != 0 && op_type != OP_GEMV) {
        cmds_.push_back(FimCommand(FimCmdType::JUMP, lc, cmds_.size() + 1));
    }

    cmds_.push_back(FimCommand(FimCmdType::EXIT, 0));

    int nop_cnt = 8 - cmds_.size() % 8;
    for (int i = 0; i < nop_cnt; i++) cmds_.push_back(FimCommand(FimCmdType::NOP, 0));

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}
void FimCrfBinGen::change_to_binary(uint8_t* crf_binary, int* crf_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    FimCommand nop_cmd(FimCmdType::NOP, 0);
    *crf_size = cmds_.size() * sizeof(uint32_t);

    for (int i = 0; i < cmds_.size(); i++) {
        uint32_t u32_data_ = cmds_[i].to_int();
        memcpy(&crf_binary[i * 4], &u32_data_, sizeof(uint32_t));
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
