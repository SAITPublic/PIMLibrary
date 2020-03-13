#ifndef _FIM_TRACE_COALESCER_H_
#define _FIM_TRACE_COALESCER_H_

#include <string.h>
#include <iostream>
#include <sstream>
#include <vector>
#include "fim_data_types.h"

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
