/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_CRF_BIN_GEN_H_
#define _PIM_CRF_BIN_GEN_H_

#include <map>
#include <vector>
#include "executor/PimCommand.h"
#include "manager/PimInfo.h"
#include "manager/PimManager.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace executor
{
class PimCrfBinGen
{
   public:
    PimCrfBinGen(pim::runtime::manager::PimManager* pim_manager);
    virtual ~PimCrfBinGen(void);

    int initialize(void);
    int deinitialize(void);
    void create_pim_cmd(PimOpType op_type, int lc);
    void change_to_binary(uint8_t* crf_binary, int* crf_size);
    void set_gemv_tile_tree(bool is_gemv_tile_tree);
    int preprocess_srf(PimBo* beta, PimBo* gamma, PimBo* mean, PimBo* variance, double epsilon, uint8_t* srf_binary);
    int get_loop_counter(PimOpType op_type, int input_size);
    void* make_crf_bin(PimOpType op_type, int data_size);
    uint8_t* find_crf(PimOpType op_type, int data_size);

   private:
    void gen_binary_with_loop(PimOpType op_type, int lc, uint8_t* bin_buf, int* crf_sz);

   private:
    pim::runtime::manager::PimManager* pim_manager_;
    std::shared_ptr<pim::runtime::manager::PimDevice> pim_device_;
    std::vector<PimCommand> cmds_;
    std::map<std::pair<PimOpType, int>, uint8_t*> crf_lut_;
    PimBlockInfo* pbi_;
    bool is_gemv_tile_tree_;
    int max_crf_size_;
    PimGemvType pim_gemv_type_;
};

} /* namespace executor */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_CRF_BIN_GEN_H_ */
