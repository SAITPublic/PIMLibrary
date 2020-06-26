#include "FimCmd.h"

namespace crfgen_offline
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

uint32_t FimCommand::to_int() const
{
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

} /* namespace crfgen_offline */
