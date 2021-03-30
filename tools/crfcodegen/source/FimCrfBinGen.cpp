/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "FimCrfBinGen.h"
#include <cmath>

namespace crfgen_offline
{
FimCrfBinGen::FimCrfBinGen() {}

void FimCrfBinGen::convert_to_binary(std::vector<FimCommand>& fim_cmd_vec, uint32_t* crf_binary)
{
    //*crf_size = fim_cmd_vec.size();

    for (int i = 0; i < fim_cmd_vec.size(); i++) {
        crf_binary[i] = fim_cmd_vec[i].to_int();
    }
}

void FimCrfBinGen::convert_to_binary(FimCommand fim_cmd, uint32_t* crf_binary) { *crf_binary = fim_cmd.to_int(); }

}  // namespace crfgen_offline
