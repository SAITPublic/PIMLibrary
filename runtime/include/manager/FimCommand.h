#ifndef _FIM_COMMAND_H_
#define _FIM_COMMAND_H_

#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
namespace manager
{
enum class FimCmdType { NOP, ADD, MUL, MAC, MAD, REV0, REV1, REV2, MOV, FILL, REV3, REV4, REV5, REV6, JUMP, EXIT };
enum class FimOpdType { A_OUT, M_OUT, EVEN_BANK, ODD_BANK, GRF_A, GRF_B, SRF_M, SRF_A };

class FimCommand
{
   public:
    FimCmdType type_;
    FimOpdType dst_;
    FimOpdType src0_;
    FimOpdType src1_;
    FimOpdType src2_;
    int loop_counter_;
    int loop_offset_;
    int is_auto_;
    int dst_idx_;
    int src0_idx_;
    int src1_idx_;
    int is_relu_;

    FimCommand()
        : type_(FimCmdType::NOP),
          dst_(FimOpdType::A_OUT),
          src0_(FimOpdType::A_OUT),
          src1_(FimOpdType::A_OUT),
          src2_(FimOpdType::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0)
    {
    }

    // NOP, CTRL
    FimCommand(FimCmdType type, int loop_counter)
        : type_(type),
          dst_(FimOpdType::A_OUT),
          src0_(FimOpdType::A_OUT),
          src1_(FimOpdType::A_OUT),
          src2_(FimOpdType::A_OUT),
          loop_counter_(loop_counter),
          loop_offset_(0),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0)

    {
    }

    // JUMP
    FimCommand(FimCmdType type, int loop_counter, int loop_offset)
        : type_(type),
          dst_(FimOpdType::A_OUT),
          src0_(FimOpdType::A_OUT),
          src1_(FimOpdType::A_OUT),
          src2_(FimOpdType::A_OUT),
          loop_counter_(loop_counter),
          loop_offset_(loop_offset),
          is_auto_(0),
          dst_idx_(0),
          src0_idx_(0),
          src1_idx_(0)
    {
    }

    FimCommand(FimCmdType type, FimOpdType dst, FimOpdType src0, int is_auto = 0, int dst_idx = 0, int src0_idx = 0,
               int src1_idx = 0, int is_relu = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(FimOpdType::A_OUT),
          src2_(FimOpdType::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(is_auto),
          dst_idx_(dst_idx),
          src0_idx_(src0_idx),
          src1_idx_(src1_idx),
          is_relu_(is_relu)
    {
    }

    FimCommand(FimCmdType type, FimOpdType dst, FimOpdType src0, FimOpdType src1, int is_auto = 0, int dst_idx = 0,
               int src0_idx = 0, int src1_idx = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(src1),
          src2_(FimOpdType::A_OUT),
          loop_counter_(0),
          loop_offset_(0),
          is_auto_(is_auto),
          dst_idx_(dst_idx),
          src0_idx_(src0_idx),
          src1_idx_(src1_idx)
    {
    }

    FimCommand(FimCmdType type, FimOpdType dst, FimOpdType src0, FimOpdType src1, FimOpdType src2, int is_auto = 0,
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
          src1_idx_(src1_idx)
    {
    }

    uint32_t bitmask(int bit) const { return (1 << bit) - 1; }
    uint32_t to_bit(uint32_t val, int bit_len, int bit_pos) const { return ((val & bitmask(bit_len)) << bit_pos); }
    uint32_t from_bit(uint32_t val, int bit_len, int bit_pos) const { return ((val >> bit_pos) & bitmask(bit_len)); }

    std::string opd_to_str(FimOpdType opd, int idx = 0) const
    {
        switch (opd) {
            case FimOpdType::A_OUT:
                return "A_OUT";
            case FimOpdType::M_OUT:
                return "M_OUT";
            case FimOpdType::EVEN_BANK:
                return "EVEN_BANK";
            case FimOpdType::ODD_BANK:
                return "ODD_BANK";
            case FimOpdType::GRF_A:
                return "GRF_A[" + std::to_string(idx) + "]";
            case FimOpdType::GRF_B:
                return "GRF_B[" + std::to_string(idx) + "]";
            case FimOpdType::SRF_M:
                return "SRF_M[" + std::to_string(idx) + "]";
            case FimOpdType::SRF_A:
                return "SRF_A[" + std::to_string(idx) + "]";
            default:
                return "NOT_DEFINED";
        }
    }

    std::string cmd_to_str(FimCmdType type) const
    {
        switch (type_) {
            case FimCmdType::EXIT:
                return "EXIT";
            case FimCmdType::NOP:
                return "NOP";
            case FimCmdType::JUMP:
                return "JUMP";
            case FimCmdType::FILL:
                return "FILL";
            case FimCmdType::MOV:
                return "MOV";
            case FimCmdType::ADD:
                return "ADD";
            case FimCmdType::MUL:
                return "MUL";
            case FimCmdType::MAC:
                return "MAC";
            case FimCmdType::MAD:
                return "MAD";
            default:
                return "NOT_DEFINED";
        }
    }

    void from_int(uint32_t val);
    uint32_t to_int() const;
    std::string to_str() const;
};

bool operator==(const FimCommand& lhs, const FimCommand& rhs);
bool operator!=(const FimCommand& lhs, const FimCommand& rhs);

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif
