/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "pim_crf_gen_api.h"
#include "PimCrfBinGen.h"
#include "PimValidationChecker.h"
using namespace crfgen_offline;

PimCrfBinGen pim_gen;
PimValidationChecker vc;

int ConvertToBinary(std::vector<PimCommand>& pim_cmd_vec, uint32_t* crf_binary)
{
    int ret = 0;

    if (ret = vc.validation_check(pim_cmd_vec)) {
        pim_gen.convert_to_binary(pim_cmd_vec, crf_binary);

        return ret;
    }

    return ret;
}

int ConvertToBinary(PimCommand pim_cmd, uint32_t* crf_binary)
{
    int ret = 0;

    if (ret = vc.check_validate_pair(pim_cmd)) {
        pim_gen.convert_to_binary(pim_cmd, crf_binary);
        return ret;
    }

    return ret;
}
