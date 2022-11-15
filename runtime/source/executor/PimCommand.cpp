/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "executor/PimCommand.h"

namespace pim
{
namespace runtime
{
namespace executor
{
bool operator==(const PimCommand& lhs, const PimCommand& rhs) { return lhs.to_int() == rhs.to_int(); }

bool operator!=(const PimCommand& lhs, const PimCommand& rhs) { return lhs.to_int() != rhs.to_int(); }

void PimCommand::from_int(uint32_t val)
{
    type_ = PimCmdType(from_bit(val, 4, 28));
    switch (type_) {
        case PimCmdType::EXIT:
            break;
        case PimCmdType::NOP:
            loop_counter_ = from_bit(val, 11, 0);
            break;
        case PimCmdType::JUMP:
            loop_counter_ = from_bit(val, 17, 11);
            loop_offset_ = from_bit(val, 11, 0);
            break;
        case PimCmdType::FILL:
        case PimCmdType::MOV:
            dst_ = PimOpdType(from_bit(val, 3, 25));
            src0_ = PimOpdType(from_bit(val, 3, 22));
            is_relu_ = from_bit(val, 1, 12);
            dst_idx_ = from_bit(val, 4, 8);
            src0_idx_ = from_bit(val, 4, 4);
            src1_idx_ = from_bit(val, 4, 0);
            break;
        case PimCmdType::MAD:
            src2_ = PimOpdType(from_bit(val, 3, 16));
        case PimCmdType::ADD:
        case PimCmdType::MUL:
        case PimCmdType::MAC:
            dst_ = PimOpdType(from_bit(val, 3, 25));
            src0_ = PimOpdType(from_bit(val, 3, 22));
            src1_ = PimOpdType(from_bit(val, 3, 19));
            is_auto_ = from_bit(val, 1, 15);
            dst_idx_ = from_bit(val, 4, 8);
            src0_idx_ = from_bit(val, 4, 4);
            src1_idx_ = from_bit(val, 4, 0);
            break;
        default:
            break;
    }
}

uint32_t PimCommand::to_int() const
{
    // validation_check();
    uint32_t val = to_bit(int(type_), 4, 28);
    switch (type_) {
        case PimCmdType::EXIT:
            break;
        case PimCmdType::NOP:
            val |= to_bit(loop_counter_, 11, 0);
            break;
        case PimCmdType::JUMP:
            val |= to_bit(loop_counter_, 17, 11);
            val |= to_bit(loop_offset_, 11, 0);
            break;
        case PimCmdType::FILL:
        case PimCmdType::MOV:
            val |= to_bit(int(dst_), 3, 25);
            val |= to_bit(int(src0_), 3, 22);
            val |= to_bit(dst_idx_, 4, 8);
            val |= to_bit(src0_idx_, 4, 4);
            val |= to_bit(src1_idx_, 4, 0);
            val |= to_bit(is_relu_, 1, 12);

            break;
        case PimCmdType::MAD:
            val |= to_bit(int(src2_), 3, 16);

        case PimCmdType::ADD:
        case PimCmdType::MUL:
        case PimCmdType::MAC:
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

std::string PimCommand::to_str() const
{
    std::stringstream ss;
    ss << cmd_to_str(type_) << " ";
    switch (type_) {
        case PimCmdType::EXIT:
            break;
        case PimCmdType::NOP:
            ss << loop_counter_ + 1 << "x";
            break;
        case PimCmdType::JUMP:
            ss << loop_counter_ << "x ";
            ss << "[PC - " << loop_offset_ << "]";
            break;
        case PimCmdType::FILL:
        case PimCmdType::MOV:
            ss << opd_to_str(dst_, dst_idx_) << ", ";
            ss << opd_to_str(src0_, src0_idx_);
            if (is_relu_) ss << ", relu";
            break;

        case PimCmdType::ADD:
        case PimCmdType::MUL:
        case PimCmdType::MAC:
            ss << opd_to_str(dst_, dst_idx_) << ", ";
            ss << opd_to_str(src0_, src0_idx_) << ", ";
            ss << opd_to_str(src1_, src1_idx_);
            break;

        case PimCmdType::MAD:
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
} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */
