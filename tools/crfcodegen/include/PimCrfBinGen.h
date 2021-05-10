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

#include "PimCmd.h"
#include "pim_data_types.h"

namespace crfgen_offline
{
class PimCrfBinGen
{
   public:
    PimCrfBinGen();
    // void test_operand_check();
    void convert_to_binary(std::vector<PimCommand>& pim_cmd_vec, uint32_t* crf_binary);
    void convert_to_binary(PimCommand pim_cmd, uint32_t* crf_binary);
    // int check_toggle_condition();
};

}  // namespace crfgen_offline

#endif /* _CRF_CODE_GENERATOR_H_ */
