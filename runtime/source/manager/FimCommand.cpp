#include "manager/FimCommand.h"

namespace fim
{
namespace runtime
{
namespace manager
{
bool operator==(const FimCommand& lhs, const FimCommand& rhs) { return lhs.to_int() == rhs.to_int(); }

bool operator!=(const FimCommand& lhs, const FimCommand& rhs) { return lhs.to_int() != rhs.to_int(); }

void FimCommand::from_int(uint32_t val)
{
    type_ = FimCmdType(from_bit(val, 4, 28));
    switch (type_) {
        case FimCmdType::EXIT:
            break;
        case FimCmdType::NOP:
            loop_counter_ = from_bit(val, 11, 0);
            break;
        case FimCmdType::JUMP:
            loop_counter_ = from_bit(val, 17, 11);
            loop_offset_ = from_bit(val, 11, 0);
            break;
        case FimCmdType::FILL:
        case FimCmdType::MOV:
            dst_ = FimOpdType(from_bit(val, 3, 25));
            src0_ = FimOpdType(from_bit(val, 3, 22));
            is_relu_ = from_bit(val, 1, 12);
            dst_idx_ = from_bit(val, 4, 8);
            src0_idx_ = from_bit(val, 4, 4);
            src1_idx_ = from_bit(val, 4, 0);
            break;
        case FimCmdType::MAD:
            src2_ = FimOpdType(from_bit(val, 3, 16));
        case FimCmdType::ADD:
        case FimCmdType::MUL:
        case FimCmdType::MAC:
            dst_ = FimOpdType(from_bit(val, 3, 25));
            src0_ = FimOpdType(from_bit(val, 3, 22));
            src1_ = FimOpdType(from_bit(val, 3, 19));
            is_auto_ = from_bit(val, 1, 15);
            dst_idx_ = from_bit(val, 4, 8);
            src0_idx_ = from_bit(val, 4, 4);
            src1_idx_ = from_bit(val, 4, 0);
            break;
        default:
            break;
    }
}

int FimCommand::is_register(FimOpdType opd_type)
{
    int is_reg = 0;
    switch (opd_type) {
        case FimOpdType::GRF_A:
        case FimOpdType::GRF_B:
        case FimOpdType::SRF_M:
        case FimOpdType::SRF_A:
            is_reg = 1;
            break;
        default:
            break;
    }

    return is_reg;
}

FimOpdTableType FimCommand::change_opd_type(FimOpdType opd_type) const
{
    FimOpdTableType return_opd_type;
    switch (opd_type) {
        case FimOpdType::EVEN_BANK:
            return_opd_type = FimOpdTableType::EVEN_BANK;
            break;
        case FimOpdType::ODD_BANK:
            return_opd_type = FimOpdTableType::ODD_BANK;
            break;
        case FimOpdType::GRF_A:
            return_opd_type = FimOpdTableType::GRF_A;
            break;
        case FimOpdType::GRF_B:
            return_opd_type = FimOpdTableType::GRF_B;
            break;
        case FimOpdType::SRF_M:
            return_opd_type = FimOpdTableType::SRF_M;
            break;
        case FimOpdType::SRF_A:
            return_opd_type = FimOpdTableType::SRF_A;
            break;
        case FimOpdType::A_OUT:
        case FimOpdType::M_OUT:
            return_opd_type = FimOpdTableType::ETC;
    }

    return return_opd_type;
}

FimCmdTableType FimCommand::change_cmd_type(FimCmdType cmd_type) const
{
    FimCmdTableType return_cmd_type;
    switch (cmd_type) {
        case FimCmdType::MUL:
            return_cmd_type = FimCmdTableType::MUL;
            break;
        case FimCmdType::ADD:
            return_cmd_type = FimCmdTableType::ADD;
            break;
        case FimCmdType::MAC:
            return_cmd_type = FimCmdTableType::MAC;
            break;
        case FimCmdType::MAD:
            return_cmd_type = FimCmdTableType::MAD;
            break;
        case FimCmdType::EXIT:
        case FimCmdType::NOP:
        case FimCmdType::JUMP:
        case FimCmdType::FILL:
        case FimCmdType::MOV:
            return_cmd_type = FimCmdTableType::ETC;
            break;

        default:
            break;
    }
    return return_cmd_type;
}

int FimCommand::validation_check(std::ofstream& src_pair_log) const
{
    int vaild_flag = 1;

    FimCmdTableType cmd_type = change_cmd_type(type_);
    int i_cmd_type = static_cast<int>(cmd_type);

    if (cmd_type != FimCmdTableType::ETC) {
        FimOpdTableType src0 = change_opd_type(src0_);
        FimOpdTableType src1 = change_opd_type(src1_);
        FimOpdTableType src2 = change_opd_type(src2_);
        int i_src0 = static_cast<int>(src0);
        int i_src1 = static_cast<int>(src1);
        int i_src2 = static_cast<int>(src2);

        if (src_pair_table[i_src0][i_src1][i_cmd_type] == 0) {
            std::cout << "Invalid in ISA 1.0  ( " << to_str() << " ) - This operation not support the current src pair."
                      << std::endl;
            vaild_flag = 0;
            //  src_pair_log << " 0 " ;
        }

        if (src_pair_table[i_src0][i_src1][i_cmd_type] == 2) {
            if (!is_auto_) {
                std::cout << "Invalid in ISA 1.0  ( " << to_str()
                          << " ) - This operation and src pair support only CAM mode" << std::endl;
                vaild_flag = 0;
                //  src_pair_log << " 0 " ;
            }
        }

        if (src_pair_table[i_src0][i_src1][i_cmd_type] == 3) {
            if (is_auto_) {
                std::cout << "Invalid in ISA 1.0  ( " << to_str()
                          << " ) - This operation and src pair not support CAM mode" << std::endl;
                vaild_flag = 0;
                // src_pair_log<< " 0 " ;
            }
        }

        // if (vaild_flag)
        //     src_pair_log << " 1 " ;

        if ((src2_ != FimOpdType::SRF_A) && (src2_ != FimOpdType::A_OUT)) {
            std::cout << "Invalid in ISA 1.0  ( " << to_str() << " ) - The src2 use only SRF_A " << std::endl;
            vaild_flag = 0;
        }

        if ((src2_ == FimOpdType::SRF_A) && (type_ != FimCmdType::MAD)) {
            std::cout << "Invalid in ISA 1.0  ( " << to_str() << " ) - src2 is used only in MAD operation" << std::endl;
            vaild_flag = 0;
        }

        if ((type_ == FimCmdType::MAC) && (dst_ == FimOpdType::GRF_A)) {
            std::cout << "Invalid in ISA 1.0  ( " << to_str() << " ) - MAC operation not use grf_a as dst" << std::endl;
            vaild_flag = 0;
        }

        if (dst_ != FimOpdType::GRF_A && dst_ != FimOpdType::GRF_B) {
            std::cout << "Invalid in ISA 1.0  ( " << to_str() << " ) - All operation use only grf_a or grf_b as dst"
                      << std::endl;
            vaild_flag = 0;
        }

        if (vaild_flag)
            src_pair_log << " 1 ";
        else {
            src_pair_log << " 0 ";
        }
    }

    if (type_ == FimCmdType::MOV || type_ == FimCmdType::FILL) {
        if (dst_ == FimOpdType::EVEN_BANK || dst_ == FimOpdType::ODD_BANK) {
            if (src0_ == FimOpdType::GRF_A || src0_ == FimOpdType::GRF_B || src1_ == FimOpdType::GRF_A ||
                src1_ == FimOpdType::GRF_B || src2_ == FimOpdType::GRF_A || src2_ == FimOpdType::GRF_B) {
                std::cerr << "ERROR) Invalid in ISA 1.0 " << to_str() << std::endl;
                exit(-1);
            }
        }
    }

    return vaild_flag;
}

uint32_t FimCommand::to_int() const
{
    // validation_check();
    uint32_t val = to_bit(int(type_), 4, 28);
    switch (type_) {
        case FimCmdType::EXIT:
            break;
        case FimCmdType::NOP:
            val |= to_bit(loop_counter_, 11, 0);
            break;
        case FimCmdType::JUMP:
            val |= to_bit(loop_counter_, 17, 11);
            val |= to_bit(loop_offset_, 11, 0);
            break;
        case FimCmdType::FILL:
        case FimCmdType::MOV:
            val |= to_bit(int(dst_), 3, 25);
            val |= to_bit(int(src0_), 3, 22);
            val |= to_bit(dst_idx_, 4, 8);
            val |= to_bit(src0_idx_, 4, 4);
            val |= to_bit(src1_idx_, 4, 0);
            val |= to_bit(is_relu_, 1, 12);

            break;
        case FimCmdType::MAD:
            val |= to_bit(int(src2_), 3, 16);

        case FimCmdType::ADD:
        case FimCmdType::MUL:
        case FimCmdType::MAC:
            val |= to_bit(int(dst_), 3, 25);
            val |= to_bit(int(src0_), 3, 22);
            val |= to_bit(int(src1_), 3, 19);
            val |= to_bit(is_auto_, 1, 15);
            val |= to_bit(dst_idx_, 4, 8);
            val |= to_bit(src0_idx_, 4, 4);
            val |= to_bit(src1_idx_, 4, 0);
            break;
        default:
            break;
    }

    return val;
}

std::string FimCommand::to_str() const
{
    std::stringstream ss;
    ss << cmd_to_str(type_) << " ";
    switch (type_) {
        case FimCmdType::EXIT:
            break;
        case FimCmdType::NOP:
            ss << loop_counter_ + 1 << "x";
            break;
        case FimCmdType::JUMP:
            ss << loop_counter_ << "x ";
            ss << "[PC - " << loop_offset_ << "]";
            break;
        case FimCmdType::FILL:
        case FimCmdType::MOV:
            ss << opd_to_str(dst_, dst_idx_) << ", ";
            ss << opd_to_str(src0_, src0_idx_);
            if (is_relu_) ss << ", relu";
            break;

        case FimCmdType::ADD:
        case FimCmdType::MUL:
        case FimCmdType::MAC:
            ss << opd_to_str(dst_, dst_idx_) << ", ";
            ss << opd_to_str(src0_, src0_idx_) << ", ";
            ss << opd_to_str(src1_, src1_idx_);
            break;

        case FimCmdType::MAD:
            ss << opd_to_str(dst_, dst_idx_) << ", ";
            ss << opd_to_str(src0_, src0_idx_) << ", ";
            ss << opd_to_str(src1_, src1_idx_) << ", ";
            ss << opd_to_str(src2_, src1_idx_);
            break;

        default:
            break;
    }
    if (is_auto_) {
        ss << ", auto";
    }
    return ss.str();
}

int FimCommand::get_opcode_idx()
{
    switch (type_) {
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

int FimCommand::is_read_register()
{
    if (is_register(src0_) || is_register(src1_) || type_ == FimCmdType::MAC || is_register(src2_)) return 1;

    return 0;
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
