/***************************************************************************************************
* Copyright (C) 2021 Samsung Electronics Co. LTD
*
* This software is a property of Samsung Electronics.
* No part of this software, either material or conceptual may be copied or distributed, transmitted,
* transcribed, stored in a retrieval system, or translated into any human or computer language in
* any form by any means,electronic, mechanical, manual or otherwise, or disclosed
* to third parties without the express written permission of Samsung Electronics.
* (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
***************************************************************************************************/

#ifndef __PIM_KERNEL_GEN_H__
#define __PIM_KERNEL_GEN_H__

#include <vector>
#include "PIMCmd.h"
#include "MultiChannelMemorySystem.h"
#include "SystemConfiguration.h"
#include "tests/KernelAddrGen.h"

using namespace std;
using namespace DRAMSim;

class IPIMKernel
{
  public:
    IPIMKernel(KernelType ktype) : kernelType(ktype) {}
    virtual vector<pimCmd> generateKernel(int num_jump_to_be_taken,
                                          int num_jump_to_be_taken_odd_bank,
                                          int num_jump_to_be_taken_even_bank) = 0;
  protected:
    KernelType kernelType;
};

/*
class BatchNormPIMKernel : public IPIMKernel
{
  public:
    BatchNormPIMKernel(KernelType ktype) : IPIMKernel(ktype) {}
    virtual vector<pimCmd> generateKernel(int num_jump_to_be_taken,
                                          int num_jump_to_be_taken_odd_bank = 0,
                                          int num_jump_to_be_taken_even_bank = 0) override
    {
        vector<pimCmd> pim_cmds;
        vector<pimCmd> tmp_cmds
        {
            pimCmd(pimCmdType::MAD, pimOpdType::GRF_A, pimOpdType::EVEN_BANK, pimOpdType::SRF_M,
                   pimOpdType::SRF_A, 1, 0, 0, 0),
            pimCmd(pimCmdType::MAD, pimOpdType::GRF_A, pimOpdType::GRF_A, pimOpdType::SRF_M,
                   pimOpdType::SRF_A, 1, 0, 0, 1),
            pimCmd(pimCmdType::NOP, 7),
            pimCmd(pimCmdType::MAD, pimOpdType::GRF_B, pimOpdType::ODD_BANK, pimOpdType::SRF_M,
                   pimOpdType::SRF_A, 1, 0, 0, 0),
            pimCmd(pimCmdType::MAD, pimOpdType::GRF_B, pimOpdType::GRF_B, pimOpdType::SRF_M,
                   pimOpdType::SRF_A, 1, 0, 0, 1),
            pimCmd(pimCmdType::NOP, 7),
            pimCmd(pimCmdType::NOP, 0)
        };
        pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
        if (num_jump_to_be_taken != 0)
        {
            pim_cmds.push_back(pimCmd(pimCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
        }
        pim_cmds.push_back(pimCmd(pimCmdType::EXIT, 0));
        return pim_cmds;
    }
};
*/
class EltwisePIMKernel : public IPIMKernel
{
  public:
    EltwisePIMKernel(KernelType ktype) : IPIMKernel(ktype) {}
    virtual vector<pimCmd> generateKernel(int num_jump_to_be_taken,
                                          int num_jump_to_be_taken_odd_bank = 0,
                                          int num_jump_to_be_taken_even_bank = 0) override
    {
        vector<pimCmd> pim_cmds;
        pimCmdType pimType = getPimCmdType();
        vector<pimCmd> tmp_cmds
        {
            pimCmd(pimCmdType::FILL, pimOpdType::GRF_A, pimOpdType::EVEN_BANK),
            pimCmd(pimType, pimOpdType::GRF_A, pimOpdType::GRF_A, pimOpdType::EVEN_BANK, 1),
            pimCmd(pimCmdType::NOP, 7),
            pimCmd(pimCmdType::FILL, pimOpdType::GRF_B, pimOpdType::ODD_BANK),
            pimCmd(pimType, pimOpdType::GRF_B, pimOpdType::GRF_B, pimOpdType::ODD_BANK, 1),
            pimCmd(pimCmdType::NOP, 7),
            pimCmd(pimCmdType::NOP, 0),
        };
        pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
        if (num_jump_to_be_taken != 0)
        {
            pim_cmds.push_back(pimCmd(pimCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
        }
        pim_cmds.push_back(pimCmd(pimCmdType::EXIT, 0));
        return pim_cmds;
    }
  private:
    pimCmdType getPimCmdType()
    {
        if (kernelType == KernelType::ADD)
            return pimCmdType::ADD;
        else if (kernelType == KernelType::MUL)
            return pimCmdType::MUL;
        else
            throw invalid_argument("Not supported element-wise operation");
    }
};

class ActPIMKernel : public IPIMKernel
{
  public:
    ActPIMKernel(KernelType ktype) : IPIMKernel(ktype) {}
    virtual vector<pimCmd> generateKernel(int num_jump_to_be_taken,
                                          int num_jump_to_be_taken_odd_bank = 0,
                                          int num_jump_to_be_taken_even_bank = 0) override
    {
        vector<pimCmd> pim_cmds;
        if (kernelType == KernelType::RELU)
        {
            vector<pimCmd> tmp_cmds
            {
                pimCmd(pimCmdType::FILL, pimOpdType::GRF_A, pimOpdType::EVEN_BANK, 1, 0, 0, 0, 1),
                pimCmd(pimCmdType::NOP, 7),
                pimCmd(pimCmdType::FILL, pimOpdType::GRF_B, pimOpdType::ODD_BANK, 1, 0, 0, 0, 1),
                pimCmd(pimCmdType::NOP, 7),
                pimCmd(pimCmdType::NOP, 0)
            };
            pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
        }
        else
        {
            throw invalid_argument("Not supported activation");
        }
        if (num_jump_to_be_taken != 0)
        {
            pim_cmds.push_back(pimCmd(pimCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
        }
        pim_cmds.push_back(pimCmd(pimCmdType::EXIT, 0));
        return pim_cmds;
    }
};

class GemvPIMKernel : public IPIMKernel
{
  public:
    GemvPIMKernel(KernelType ktype) : IPIMKernel(ktype) {}
    virtual vector<pimCmd> generateKernel(int num_jump_to_be_taken,
                                          int num_jump_to_be_taken_odd_bank,
                                          int num_jump_to_be_taken_even_bank) override
    {
        vector<pimCmd> pim_cmds;
        if (kernelType == KernelType::GEMV)
        {
            vector<pimCmd> tmp_cmds
            {
                pimCmd(pimCmdType::MAC, pimOpdType::GRF_B, pimOpdType::GRF_A,
                       pimOpdType::EVEN_BANK, 1, 0, 0, 0),
                pimCmd(pimCmdType::JUMP, num_jump_to_be_taken_even_bank, 2),
                pimCmd(pimCmdType::MAC, pimOpdType::GRF_B, pimOpdType::GRF_A,
                       pimOpdType::ODD_BANK , 1, 0, 0, 0),
                pimCmd(pimCmdType::JUMP, num_jump_to_be_taken_odd_bank, 2),
                pimCmd(pimCmdType::NOP, 7),
            };
            pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
        }
        else if (kernelType == KernelType::GEMVTREE)
        {
            vector<pimCmd> tmp_cmds
            {
                pimCmd(pimCmdType::MAC, pimOpdType::GRF_B, pimOpdType::GRF_A,
                       pimOpdType::EVEN_BANK, 1, 0, 0, 0),
                // FIXME: hard coding
                pimCmd(pimCmdType::JUMP, 7, 2),
                pimCmd(pimCmdType::NOP, 7),
                pimCmd(pimCmdType::MUL, pimOpdType::GRF_B, pimOpdType::GRF_B,
                       pimOpdType::EVEN_BANK, 1),
                pimCmd(pimCmdType::MAC, pimOpdType::GRF_B, pimOpdType::GRF_A,
                       pimOpdType::ODD_BANK, 1, 0, 0, 0),
                pimCmd(pimCmdType::JUMP, 7, 2),
                pimCmd(pimCmdType::NOP, 7),
                pimCmd(pimCmdType::MUL, pimOpdType::GRF_B, pimOpdType::GRF_B,
                       pimOpdType::EVEN_BANK, 1),
                //pimCmd(pimCmdType::JUMP, num_jump, 7), /*it used that tile is 2*/
            };
            pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
        }
        else
        {
            throw invalid_argument("Not supported gemv operation");
        }
        if (num_jump_to_be_taken != 0)
        {
            pim_cmds.push_back(pimCmd(pimCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
        }
        pim_cmds.push_back(pimCmd(pimCmdType::EXIT, 0));
        return pim_cmds;
    }
};

class PIMKernelGen
{
  public:
      static vector<pimCmd> getPimCmds(KernelType ktype, int num_jump_to_be_taken,
                                       int num_jump_to_be_taken_odd_bank,
                                       int num_jump_to_be_taken_even_bank);
};

#endif // __PIM_KERNEL_GEN_H__
