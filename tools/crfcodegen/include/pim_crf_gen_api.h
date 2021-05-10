/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_CRF_GEN_API_H_
#define _PIM_CRF_GEN_API_H_
#include <vector>
#include "PimCmd.h"

/** @file pim_crf_gen_api.h
 *  @date 2020.06.26
 *  @author Seungwoo Seo <sgwoo.seo@samsung.com>
 *  @brief PIM CRF Binary offline tool Documentation
 */

using namespace crfgen_offline;

/**
 * @defgroup PIM-Binary-Gen-API-Documentation "PIM CRF Binary Gen Documentation"
 * @{
 */

/**
 * @brief Convert PIM command vector to binary.
 *
 * @param pim_cmd_vec [in] pim command vector
 * @param crf_binary  [out] converted binary array
 *
 * @return Return success/failure
 */

int ConvertToBinary(std::vector<PimCommand>& pim_cmd_vec, uint32_t* crf_binary);

/**
 * @brief Convert PIM command to binary.
 *
 * @param pim_cmd     [in]  pim command
 * @param crf_binary  [out] converted binary
 *
 * @return Return success/failure
 */
int ConvertToBinary(PimCommand pim_cmd, uint32_t* crf_binary);

#endif
