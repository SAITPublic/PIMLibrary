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

#include <vector>
#include "manager/PimCommand.h"
#include "manager/PimInfo.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class PimCrfBinGen
{
   public:
    PimCrfBinGen();

    void create_pim_cmd(PimOpType op_type, int lc);
    void change_to_binary(uint8_t* crf_binary, int* crf_size);
    void gen_binary_with_loop(PimOpType op_type, int lc, uint8_t* bin_buf, int* crf_sz);
    void set_gemv_tile_tree(bool is_gemv_tile_tree);

   private:
    std::vector<PimCommand> cmds_;
    bool is_gemv_tile_tree_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */

#endif /* _CRF_CODE_GENERATOR_H_ */
