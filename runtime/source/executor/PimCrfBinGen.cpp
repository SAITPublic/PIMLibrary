/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "executor/PimCrfBinGen.h"
#include <cmath>

namespace pim
{
namespace runtime
{
namespace executor
{
PimCrfBinGen::PimCrfBinGen(pim::runtime::manager::PimManager* pim_manager)
    : pim_manager_(pim_manager), is_gemv_tile_tree_(true), max_crf_size_(128)
{
    pim_device_ = pim_manager_->get_pim_device();
    pbi_ = pim_device_->get_pim_block_info();
}

PimCrfBinGen::~PimCrfBinGen(void)
{
    for (auto it = crf_lut_.begin(); it != crf_lut_.end(); it++) {
        pim_manager_->free_memory((void*)it->second, MEM_TYPE_DEVICE);
    }
    crf_lut_.clear();
    pim_device_.reset();
}

void PimCrfBinGen::gen_binary_with_loop(PimOpType op_type, int lc, uint8_t* bin_buf, int* crf_sz)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    create_pim_cmd(op_type, lc);
    change_to_binary(bin_buf, crf_sz);
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

void PimCrfBinGen::create_pim_cmd(PimOpType op_type, int lc)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    if (op_type == OP_ELT_ADD) {
        std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
            PimCommand(PimCmdType::ADD, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1),
            PimCommand(PimCmdType::NOP, 23), PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK),
            PimCommand(PimCmdType::ADD, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1),
            PimCommand(PimCmdType::NOP, 23) /*,
            PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_ELT_MUL) {
        std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
            PimCommand(PimCmdType::MUL, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1),
            PimCommand(PimCmdType::NOP, 23), PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK),
            PimCommand(PimCmdType::MUL, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1),
            PimCommand(PimCmdType::NOP, 23) /*,
            PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_RELU) {
        std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0, 1),
            PimCommand(PimCmdType::NOP, 15),
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1, 0, 0, 0, 1),
            PimCommand(PimCmdType::NOP, 15) /*, PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_COPY) {
        std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0, 0),
            PimCommand(PimCmdType::NOP, 15),
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1, 0, 0, 0, 0),
            PimCommand(PimCmdType::NOP, 15) /*, PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else if (op_type == OP_GEMV) {
        if (is_gemv_tile_tree_) {
            std::vector<PimCommand> tmp_cmds{
                PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0),
                PimCommand(PimCmdType::JUMP, 7, 2),
                PimCommand(PimCmdType::NOP, 23),
                PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::ODD_BANK, 1, 0, 0, 0),
                PimCommand(PimCmdType::JUMP, 7, 2),
                PimCommand(PimCmdType::NOP, 23),
            };
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
        } else {
            int even_lc = 8 * ceil((float)lc / 2) - 1;
            int odd_lc = 8 * (lc / 2) - 1;
            std::vector<PimCommand> tmp_cmds{
                PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0),
                PimCommand(PimCmdType::JUMP, even_lc, 2),
                PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::ODD_BANK, 1, 0, 0, 0),
                PimCommand(PimCmdType::JUMP, odd_lc, 2), PimCommand(PimCmdType::NOP, 23)};
            cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
        }
    } else if (op_type == OP_BN) {
        std::vector<PimCommand> tmp_cmds{PimCommand(PimCmdType::MAD, PimOpdType::GRF_A, PimOpdType::EVEN_BANK,
                                                    PimOpdType::SRF_M, PimOpdType::SRF_A, 1, 0, 0, 0),
                                         PimCommand(PimCmdType::NOP, 7),
                                         PimCommand(PimCmdType::MAD, PimOpdType::GRF_A, PimOpdType::GRF_A,
                                                    PimOpdType::SRF_M, PimOpdType::SRF_A, 1, 0, 0, 1),
                                         PimCommand(PimCmdType::NOP, 7),
                                         PimCommand(PimCmdType::MAD, PimOpdType::GRF_B, PimOpdType::ODD_BANK,
                                                    PimOpdType::SRF_M, PimOpdType::SRF_A, 1, 0, 0, 0),
                                         PimCommand(PimCmdType::NOP, 7),
                                         PimCommand(PimCmdType::MAD, PimOpdType::GRF_B, PimOpdType::GRF_B,
                                                    PimOpdType::SRF_M, PimOpdType::SRF_A, 1, 0, 0, 1),
                                         PimCommand(PimCmdType::NOP, 23)
                                         /*PimCommand(PimCmdType::NOP, 0)*/};
        cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    }

    if ((lc != 0 && is_gemv_tile_tree_) == true || op_type != OP_GEMV) {
        cmds_.push_back(PimCommand(PimCmdType::JUMP, lc, cmds_.size() + 1));
    }

    cmds_.push_back(PimCommand(PimCmdType::EXIT, 0));

    int nop_cnt = 8 - cmds_.size() % 8;
    for (int i = 0; i < nop_cnt; i++) cmds_.push_back(PimCommand(PimCmdType::NOP, 0));

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

void PimCrfBinGen::change_to_binary(uint8_t* crf_binary, int* crf_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    PimCommand nop_cmd(PimCmdType::NOP, 0);
    *crf_size = cmds_.size() * sizeof(uint32_t);

    for (int i = 0; i < cmds_.size(); i++) {
        uint32_t u32_data_ = cmds_[i].to_int();
        memcpy(&crf_binary[i * 4], &u32_data_, sizeof(uint32_t));
    }
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

void PimCrfBinGen::set_gemv_tile_tree(bool is_gemv_tile_tree) { is_gemv_tile_tree_ = is_gemv_tile_tree; }
int PimCrfBinGen::preprocess_srf(PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance, double epsilon,
                                 uint8_t* srf_binary)
{
    int num_pim_rank = pbi_->num_pim_rank;
    int num_pim_chan = pbi_->num_pim_chan;

    int cidx = 0;
    int rank = 0;
    int burst_idx = 0;
    int num_stride_reg = 2;
    int num_half_per_reg = 16;

    half* h_srf_binary = reinterpret_cast<half*>(srf_binary);
    half* h_beta = (half*)beta->data;
    half* h_gamma = (half*)gamma->data;
    half* h_var = (half*)variance->data;
    half* h_mean = (half*)mean->data;

    for (int ch_model = 0; ch_model < beta->bshape.c; ch_model++) {
        h_srf_binary[cidx * num_pim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx] =
            1 / sqrt((float)h_var[ch_model] + epsilon);  // scale
        h_srf_binary[cidx * num_pim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 1] =
            h_gamma[ch_model];  // gamma
        h_srf_binary[cidx * num_pim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 8] =
            -(float)h_mean[ch_model] / sqrt((float)h_var[ch_model] + epsilon);  // shift
        h_srf_binary[cidx * num_pim_rank * num_half_per_reg + rank * num_half_per_reg + burst_idx + 9] =
            h_beta[ch_model];  // beta
        rank++;

        if (rank >= num_pim_rank) {
            rank = 0;
            cidx++;
        }
        if (cidx >= num_pim_chan) {
            cidx = 0;
            burst_idx += num_stride_reg;
        }
        if (burst_idx >= 8) {
            std::cout << "error: this is not defined" << std::endl;
        }
    }
    return 0;
}

int PimCrfBinGen::get_loop_counter(PimOpType op_type, int input_size)
{
    int lc = 0;
    int num_transaction = (input_size / 16) / sizeof(uint16_t);
    int num_parallelism = pbi_->num_pim_blocks * pbi_->num_pim_chan * pbi_->num_pim_rank * pbi_->num_grf;
    int num_tile = num_transaction / num_parallelism;

    if (op_type == OP_GEMV) {
        if (pim_gemv_type_ == TILE_TREE)
            lc = (input_size / pbi_->trans_size / pbi_->num_grf_A / 2) - 1;
        else
            lc = input_size / pbi_->trans_size / pbi_->num_grf_A;
    } else
        lc = num_tile / 2 - 1;
    return lc;
}

void* PimCrfBinGen::make_crf_bin(PimOpType op_type, int data_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    uint8_t* h_crf = new uint8_t[max_crf_size_];
    uint8_t* d_crf;
    int crf_size;
    pim_manager_->alloc_memory((void**)&d_crf, max_crf_size_, MEM_TYPE_DEVICE);

    int lc = get_loop_counter(op_type, data_size);
    gen_binary_with_loop(op_type, lc, h_crf, &crf_size);

    pim_manager_->copy_memory((void*)d_crf, (void*)h_crf, max_crf_size_, HOST_TO_DEVICE);
    crf_lut_.insert(std::make_pair(std::make_pair(op_type, data_size), d_crf));
    delete[] h_crf;

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return (void*)d_crf;
}

uint8_t* PimCrfBinGen::find_crf(PimOpType op_type, int data_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    uint8_t* addr = nullptr;

    std::map<std::pair<PimOpType, int>, uint8_t*>::const_iterator found =
        crf_lut_.find(std::make_pair(op_type, data_size));
    if (found != crf_lut_.end()) {
        addr = found->second;
    }

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return addr;
}
} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */
