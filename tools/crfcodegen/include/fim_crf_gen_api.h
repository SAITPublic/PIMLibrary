#ifndef _FIM_CRF_GEN_API_H_
#define _FIM_CRF_GEN_API_H_
#include <vector>
#include "FimCmd.h"

/** @file fim_crf_gen_api.h
 *  @date 2020.06.26
 *  @author Seungwoo Seo <sgwoo.seo@samsung.com>
 *  @brief FIM CRF Binary offline tool Documentation
 */

using namespace crfgen_offline;

/**
 * @defgroup FIM-Binary-Gen-API-Documentation "FIM CRF Binary Gen Documentation"
 * @{
 */

/**
 * @brief Convert FIM command vector to binary.
 *
 * @param fim_cmd_vec [in] fim command vector
 * @param crf_binary  [out] converted binary array
 *
 * @return Return success/failure
 */

int ConvertToBinary(std::vector<FimCommand>& fim_cmd_vec, uint32_t* crf_binary);

/**
 * @brief Convert FIM command to binary.
 *
 * @param fim_cmd     [in]  fim command
 * @param crf_binary  [out] converted binary
 *
 * @return Return success/failure
 */
int ConvertToBinary(FimCommand fim_cmd, uint32_t* crf_binary);

#endif
