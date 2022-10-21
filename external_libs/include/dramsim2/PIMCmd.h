#ifndef __PIM_CMD_HPP__
#define __PIM_CMD_HPP__

#include <bitset>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace DRAMSim
{
enum class pimCmdType { NOP, ADD, MUL, MAC, MAD, REV0, REV1, REV2, MOV, FILL, REV3, REV4, REV5, REV6, JUMP, EXIT };

enum class pimOpdType { A_OUT, M_OUT, EVEN_BANK, ODD_BANK, GRF_A, GRF_B, SRF_M, SRF_A };

class pimCmd
{
   public:
    pimCmdType type_;
    pimOpdType dst_;
    pimOpdType src0_;
    pimOpdType src1_;
    pimOpdType src2_;
    int loopCounter_;
    int loopOffset_;
    int isAuto_;
    int dstIdx_;
    int src0Idx_;
    int src1Idx_;
    int isRelu_;

    pimCmd()
        : type_(pimCmdType::NOP),
          dst_(pimOpdType::A_OUT),
          src0_(pimOpdType::A_OUT),
          src1_(pimOpdType::A_OUT),
          src2_(pimOpdType::A_OUT),
          loopCounter_(0),
          loopOffset_(0),
          isAuto_(0),
          dstIdx_(0),
          src0Idx_(0),
          src1Idx_(0)
    {
    }

    // NOP, CTRL
    pimCmd(pimCmdType type, int loopCounter)
        : type_(type),
          dst_(pimOpdType::A_OUT),
          src0_(pimOpdType::A_OUT),
          src1_(pimOpdType::A_OUT),
          src2_(pimOpdType::A_OUT),
          loopCounter_(loopCounter),
          loopOffset_(0),
          isAuto_(0),
          dstIdx_(0),
          src0Idx_(0),
          src1Idx_(0)
    {
    }

    // JUMP
    pimCmd(pimCmdType type, int loopCounter, int loop_offset)
        : type_(type),
          dst_(pimOpdType::A_OUT),
          src0_(pimOpdType::A_OUT),
          src1_(pimOpdType::A_OUT),
          src2_(pimOpdType::A_OUT),
          loopCounter_(loopCounter),
          loopOffset_(loop_offset),
          isAuto_(0),
          dstIdx_(0),
          src0Idx_(0),
          src1Idx_(0)
    {
    }

    pimCmd(pimCmdType type, pimOpdType dst, pimOpdType src0, int is_auto = 0, int dst_idx = 0, int src0_idx = 0,
           int src1_idx = 0, int is_relu = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(pimOpdType::A_OUT),
          src2_(pimOpdType::A_OUT),
          loopCounter_(0),
          loopOffset_(0),
          isAuto_(is_auto),
          dstIdx_(dst_idx),
          src0Idx_(src0_idx),
          src1Idx_(src1_idx),
          isRelu_(is_relu)
    {
    }

    pimCmd(pimCmdType type, pimOpdType dst, pimOpdType src0, pimOpdType src1, int is_auto = 0, int dst_idx = 0,
           int src0_idx = 0, int src1_idx = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(src1),
          src2_(pimOpdType::A_OUT),
          loopCounter_(0),
          loopOffset_(0),
          isAuto_(is_auto),
          dstIdx_(dst_idx),
          src0Idx_(src0_idx),
          src1Idx_(src1_idx)
    {
    }

    pimCmd(pimCmdType type, pimOpdType dst, pimOpdType src0, pimOpdType src1, pimOpdType src2, int is_auto = 0,
           int dst_idx = 0, int src0_idx = 0, int src1_idx = 0)
        : type_(type),
          dst_(dst),
          src0_(src0),
          src1_(src1),
          src2_(src2),
          loopCounter_(0),
          loopOffset_(0),
          isAuto_(is_auto),
          dstIdx_(dst_idx),
          src0Idx_(src0_idx),
          src1Idx_(src1_idx)
    {
    }

    uint32_t bitmask(int bit) const { return (1 << bit) - 1; }
    uint32_t toBit(uint32_t val, int bit_len, int bit_pos) const { return ((val & bitmask(bit_len)) << bit_pos); }
    uint32_t fromBit(uint32_t val, int bit_len, int bit_pos) const { return ((val >> bit_pos) & bitmask(bit_len)); }

    std::string opdToStr(pimOpdType opd, int idx = 0) const
    {
        switch (opd) {
            case pimOpdType::A_OUT:
                return "A_OUT";
            case pimOpdType::M_OUT:
                return "M_OUT";
            case pimOpdType::EVEN_BANK:
                return "EVEN_BANK";
            case pimOpdType::ODD_BANK:
                return "ODD_BANK";
            case pimOpdType::GRF_A:
                return "GRF_A[" + to_string(idx) + "]";
            case pimOpdType::GRF_B:
                return "GRF_B[" + to_string(idx) + "]";
            case pimOpdType::SRF_M:
                return "SRF_M[" + to_string(idx) + "]";
            case pimOpdType::SRF_A:
                return "SRF_A[" + to_string(idx) + "]";
            default:
                return "NOT_DEFINED";
        }
    }

    std::string cmdToStr(pimCmdType type) const
    {
        switch (type_) {
            case pimCmdType::EXIT:
                return "EXIT";
            case pimCmdType::NOP:
                return "NOP";
            case pimCmdType::JUMP:
                return "JUMP";
            case pimCmdType::FILL:
                return "FILL";
            case pimCmdType::MOV:
                return "MOV";
            case pimCmdType::ADD:
                return "ADD";
            case pimCmdType::MUL:
                return "MUL";
            case pimCmdType::MAC:
                return "MAC";
            case pimCmdType::MAD:
                return "MAD";
            default:
                return "NOT_DEFINED";
        }
    }

    void fromInt(uint32_t val);
    void validationCheck() const;
    uint32_t toInt() const;
    std::string toStr() const;
};

bool operator==(const pimCmd& lhs, const pimCmd& rhs);
bool operator!=(const pimCmd& lhs, const pimCmd& rhs);

}  // namespace DRAMSim

#endif
