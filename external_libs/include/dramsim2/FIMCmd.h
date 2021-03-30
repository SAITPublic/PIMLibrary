/*********************************************************************************
 *  Copyright (c) 2010-2011, Elliott Cooper-Balis
 *                             Paul Rosenfeld
 *                             Bruce Jacob
 *                             University of Maryland
 *                             dramninjas [at] gmail [dot] com
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *notice,
 *        this list of conditions and the following disclaimer in the
 *documentation
 *        and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/
#ifndef __FIM_CMD_HPP__
#define __FIM_CMD_HPP__

#include <bitset>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace DRAMSim
{
enum class fim_cmd_type { NOP, ADD, MUL, MAC, MAD, REV0, REV1, REV2, MOV, FILL, REV3, REV4, REV5, REV6, JUMP, EXIT };

enum class fim_opd_type { A_OUT, M_OUT, EVEN_BANK, ODD_BANK, GRF_A, GRF_B, SRF_M, SRF_A };

class fim_cmd
{
   public:
    fim_cmd_type type_;
    fim_opd_type dst_;
    fim_opd_type src0_;
    fim_opd_type src1_;
    fim_opd_type src2_;
    int loop_counter_;
    int loop_offset_;
    int is_auto_;
    int dst_idx_;
    int src0_idx_;
    int src1_idx_;
    int is_relu_;

    fim_cmd()
        : type_(fim_cmd_type::NOP),
          dst_(fim_opd_type::A_OUT),
          src0_(fim_opd_type::A_OUT),
          src1_(fim_opd_type::A_OUT),
          src2_(fim_opd_type::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0)
    {
    }

    // NOP, CTRL
    fim_cmd(fim_cmd_type type, int loop_counter)
        : type_(type),
          dst_(fim_opd_type::A_OUT),
          src0_(fim_opd_type::A_OUT),
          src1_(fim_opd_type::A_OUT),
          src2_(fim_opd_type::A_OUT),
          loop_counter_(loop_counter),
          loop_offset_(0),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0)

    {
    }

    // JUMP
    fim_cmd(fim_cmd_type type, int loop_counter, int loop_offset)
        : type_(type),
          dst_(fim_opd_type::A_OUT),
          src0_(fim_opd_type::A_OUT),
          src1_(fim_opd_type::A_OUT),
          src2_(fim_opd_type::A_OUT),
          loop_counter_(loop_counter),
          loop_offset_(loop_offset),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0)
    {
    }

    fim_cmd(fim_cmd_type type, fim_opd_type dst, fim_opd_type src0, int is_auto = 0, int dst_idx = 0, int src0_idx = 0,
            int src1_idx = 0, int is_relu = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(fim_opd_type::A_OUT),
          src2_(fim_opd_type::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(is_auto),
          dst_idx_(dst_idx),
          src0_idx_(src0_idx),
          src1_idx_(src1_idx),
          is_relu_(is_relu)
    {
    }

    fim_cmd(fim_cmd_type type, fim_opd_type dst, fim_opd_type src0, fim_opd_type src1, int is_auto = 0, int dst_idx = 0,
            int src0_idx = 0, int src1_idx = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(src1),
          src2_(fim_opd_type::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(is_auto),
          dst_idx_(dst_idx),
          src0_idx_(src0_idx),
          src1_idx_(src1_idx)
    {
    }

    fim_cmd(fim_cmd_type type, fim_opd_type dst, fim_opd_type src0, fim_opd_type src1, fim_opd_type src2,
            int is_auto = 0, int dst_idx = 0, int src0_idx = 0, int src1_idx = 0)
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
          src1_idx_(src1_idx)
    {
    }

    uint32_t bitmask(int bit) const { return (1 << bit) - 1; }
    uint32_t to_bit(uint32_t val, int bit_len, int bit_pos) const { return ((val & bitmask(bit_len)) << bit_pos); }
    uint32_t from_bit(uint32_t val, int bit_len, int bit_pos) const { return ((val >> bit_pos) & bitmask(bit_len)); }

    std::string opd_to_str(fim_opd_type opd, int idx = 0) const
    {
        switch (opd) {
            case fim_opd_type::A_OUT:
                return "A_OUT";
            case fim_opd_type::M_OUT:
                return "M_OUT";
            case fim_opd_type::EVEN_BANK:
                return "EVEN_BANK";
            case fim_opd_type::ODD_BANK:
                return "ODD_BANK";
            case fim_opd_type::GRF_A:
                return "GRF_A[" + to_string(idx) + "]";
            case fim_opd_type::GRF_B:
                return "GRF_B[" + to_string(idx) + "]";
            case fim_opd_type::SRF_M:
                return "SRF_M[" + to_string(idx) + "]";
            case fim_opd_type::SRF_A:
                return "SRF_A[" + to_string(idx) + "]";
            default:
                return "NOT_DEFINED";
        }
    }

    std::string cmd_to_str(fim_cmd_type type) const
    {
        switch (type_) {
            case fim_cmd_type::EXIT:
                return "EXIT";
            case fim_cmd_type::NOP:
                return "NOP";
            case fim_cmd_type::JUMP:
                return "JUMP";
            case fim_cmd_type::FILL:
                return "FILL";
            case fim_cmd_type::MOV:
                return "MOV";
            case fim_cmd_type::ADD:
                return "ADD";
            case fim_cmd_type::MUL:
                return "MUL";
            case fim_cmd_type::MAC:
                return "MAC";
            case fim_cmd_type::MAD:
                return "MAD";
            default:
                return "NOT_DEFINED";
        }
    }

    void from_int(uint32_t val);
    void validation_check() const;
    uint32_t to_int() const;
    std::string to_str() const;
};

bool operator==(const fim_cmd& lhs, const fim_cmd& rhs);
bool operator!=(const fim_cmd& lhs, const fim_cmd& rhs);

}  // namespace DRAMSim

#endif
