#include "emulator/FimTraceCoalescer.h"

namespace fim
{
namespace runtime
{
namespace emulator
{
void TraceParser::append_data(uint8_t *dst, uint8_t *src, int size) { memcpy(dst, src, size); }
inline void coalesced_update(FimMemTraceData *fmtd32, int &coalesced_trace_it, FimMemTraceData *fmtd16, int &trace_it,
                             char prev_cmd, uint64_t &prev_addr)
{
    fmtd32[coalesced_trace_it].cmd = prev_cmd;
    fmtd32[coalesced_trace_it].block_id = fmtd16[trace_it].block_id;
    fmtd32[coalesced_trace_it].thread_id = fmtd16[trace_it].thread_id;
    fmtd32[coalesced_trace_it].addr = fmtd16[trace_it].addr;
    coalesced_trace_it++;
    prev_addr = fmtd16[trace_it].addr;
}

void TraceParser::coalesce_trace(FimMemTraceData *fmtd32, int *fmtd32_size, FimMemTraceData *fmtd16, int fmtd16_size)
{
    uint64_t prev_addr = -1;
    char prev_cmd = 'B';
    int coalesced_trace_it = 0;
    for (int trace_it = 0; trace_it < fmtd16_size; trace_it++) {
        switch (fmtd16[trace_it].cmd) {
            case 'B':
                fmtd32[coalesced_trace_it].cmd = 'B';
                fmtd32[coalesced_trace_it].block_id = fmtd16[trace_it].block_id;
                fmtd32[coalesced_trace_it].thread_id = fmtd16[trace_it].thread_id;
                fmtd32[coalesced_trace_it].addr = 0;
                coalesced_trace_it++;
                prev_addr = -1;
                prev_cmd = 'B';
                break;
            case 'R':
                if (prev_addr != -1 && prev_cmd == 'R') {
                    if ((prev_addr + TRANS_SIZE) == fmtd16[trace_it].addr)
                        ;
                    else
                        coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'R', prev_addr);
                } else {
                    coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'R', prev_addr);
                }
                prev_cmd = 'R';
                break;
            case 'O':
                if (prev_addr != -1 && prev_cmd == 'O') {
                    if ((prev_addr + TRANS_SIZE) == fmtd16[trace_it].addr)
                        ;
                    else
                        coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'O', prev_addr);
                } else {
                    coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'O', prev_addr);
                }
                prev_cmd = 'O';
                break;
            case 'W':
                if (prev_addr != -1 && prev_cmd == 'W') {
                    if ((prev_addr + TRANS_SIZE) == fmtd16[trace_it].addr) {
                        append_data(fmtd32[coalesced_trace_it - 1].data + 16, fmtd16[trace_it].data, 16);
                    } else {
                        memcpy(fmtd32[coalesced_trace_it].data, fmtd16[trace_it].data, 16);
                        coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'W', prev_addr);
                    }
                } else {
                    memcpy(fmtd32[coalesced_trace_it].data, fmtd16[trace_it].data, 16);
                    coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'W', prev_addr);
                }
                prev_cmd = 'W';
                break;
        }
    }
    fmtd32_size[0] = coalesced_trace_it;
}

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */
