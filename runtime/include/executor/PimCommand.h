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

#ifndef _PIM_COMMAND_H_
#define _PIM_COMMAND_H_

#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "utility/pim_log.h"

namespace pim
{
namespace runtime
{
namespace executor
{
enum class PimCmdType { NOP, ADD, MUL, MAC, MAD, REV0, REV1, REV2, MOV, FILL, REV3, REV4, REV5, REV6, JUMP, EXIT };
enum class PimOpdType { A_OUT, M_OUT, EVEN_BANK, ODD_BANK, GRF_A, GRF_B, SRF_M, SRF_A };

class PimCommand
{
   public:
    PimCmdType type_;
    PimOpdType dst_;
    PimOpdType src0_;
    PimOpdType src1_;
    PimOpdType src2_;
    int loop_counter_;
    int loop_offset_;
    int is_auto_;
    int dst_idx_;
    int src0_idx_;
    int src1_idx_;
    int is_relu_;

    PimCommand()
        : type_(PimCmdType::NOP),
          dst_(PimOpdType::A_OUT),
          src0_(PimOpdType::A_OUT),
          src1_(PimOpdType::A_OUT),
          src2_(PimOpdType::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0),
          is_relu_(0)
    {
    }

    // NOP, CTRL
    PimCommand(PimCmdType type, int loop_counter)
        : type_(type),
          dst_(PimOpdType::A_OUT),
          src0_(PimOpdType::A_OUT),
          src1_(PimOpdType::A_OUT),
          src2_(PimOpdType::A_OUT),
          loop_counter_(loop_counter),
          loop_offset_(0),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0),
          is_relu_(0)

    {
    }

    // JUMP
    PimCommand(PimCmdType type, int loop_counter, int loop_offset)
        : type_(type),
          dst_(PimOpdType::A_OUT),
          src0_(PimOpdType::A_OUT),
          src1_(PimOpdType::A_OUT),
          src2_(PimOpdType::A_OUT),
          loop_counter_(loop_counter),
          loop_offset_(loop_offset),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0),
          is_relu_(0)
    {
    }

    PimCommand(PimCmdType type, PimOpdType dst, PimOpdType src0, int is_auto = 0, int dst_idx = 0, int src0_idx = 0,
               int src1_idx = 0, int is_relu = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(PimOpdType::A_OUT),
          src2_(PimOpdType::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(is_auto),
          dst_idx_(dst_idx),
          src0_idx_(src0_idx),
          src1_idx_(src1_idx),
          is_relu_(is_relu)
    {
    }

    PimCommand(PimCmdType type, PimOpdType dst, PimOpdType src0, PimOpdType src1, int is_auto = 0, int dst_idx = 0,
               int src0_idx = 0, int src1_idx = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(src1),
          src2_(PimOpdType::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(is_auto),
          dst_idx_(dst_idx),
          src0_idx_(src0_idx),
          src1_idx_(src1_idx),
          is_relu_(0)
    {
    }

    PimCommand(PimCmdType type, PimOpdType dst, PimOpdType src0, PimOpdType src1, PimOpdType src2, int is_auto = 0,
               int dst_idx = 0, int src0_idx = 0, int src1_idx = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(src1),
          src2_(src2),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(is_auto),
          dst_idx_(dst_idx),
          src0_idx_(src0_idx),
          src1_idx_(src1_idx),
          is_relu_(0)
    {
    }

    uint32_t bitmask(int bit) const { return (1 << bit) - 1; }
    uint32_t to_bit(uint32_t val, int bit_len, int bit_pos) const { return ((val & bitmask(bit_len)) << bit_pos); }
    uint32_t from_bit(uint32_t val, int bit_len, int bit_pos) const { return ((val >> bit_pos) & bitmask(bit_len)); }

    std::string opd_to_str(PimOpdType opd, int idx = 0) const
    {
        switch (opd) {
            case PimOpdType::A_OUT:
                return "A_OUT";
            case PimOpdType::M_OUT:
                return "M_OUT";
            case PimOpdType::EVEN_BANK:
                return "EVEN_BANK";
            case PimOpdType::ODD_BANK:
                return "ODD_BANK";
            case PimOpdType::GRF_A:
                return "GRF_A[" + std::to_string(idx) + "]";
            case PimOpdType::GRF_B:
                return "GRF_B[" + std::to_string(idx) + "]";
            case PimOpdType::SRF_M:
                return "SRF_M[" + std::to_string(idx) + "]";
            case PimOpdType::SRF_A:
                return "SRF_A[" + std::to_string(idx) + "]";
            default:
                return "NOT_DEFINED";
        }
    }

    std::string cmd_to_str(PimCmdType type) const
    {
        switch (type_) {
            case PimCmdType::EXIT:
                return "EXIT";
            case PimCmdType::NOP:
                return "NOP";
            case PimCmdType::JUMP:
                return "JUMP";
            case PimCmdType::FILL:
                return "FILL";
            case PimCmdType::MOV:
                return "MOV";
            case PimCmdType::ADD:
                return "ADD";
            case PimCmdType::MUL:
                return "MUL";
            case PimCmdType::MAC:
                return "MAC";
            case PimCmdType::MAD:
                return "MAD";
            default:
                return "NOT_DEFINED";
        }
    }

    void from_int(uint32_t val);
    uint32_t to_int() const;
    std::string to_str() const;
};

bool operator==(const PimCommand& lhs, const PimCommand& rhs);
bool operator!=(const PimCommand& lhs, const PimCommand& rhs);

} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */

#endif
