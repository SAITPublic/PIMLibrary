/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_TRACE_COALESCER_H_
#define _FIM_TRACE_COALESCER_H_

#include <string.h>
#include <iostream>
#include <sstream>
#include <vector>
#include "fim_data_types.h"
#include "manager/FimInfo.h"
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
namespace emulator
{
// TODO: Define Stride for each FIM configuration
#define TRANS_SIZE 0x10
uint64_t MASK = (~((TRANS_SIZE << 1) - 1));

class TraceParser
{
   public:
    void coalesce_trace(FimMemTraceData *fmtd32, int *fmtd32_size, FimMemTraceData *fmtd16, int fmtd16_size);

   private:
    void append_data(uint8_t *dst, uint8_t *src, int size);
    void move_data(uint8_t *dst, uint8_t *src, int size);
};

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */

#endif
