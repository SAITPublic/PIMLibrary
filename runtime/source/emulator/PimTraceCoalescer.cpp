/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "emulator/PimTraceCoalescer.h"

namespace pim
{
namespace runtime
{
namespace emulator
{
void TraceParser::append_data(uint8_t *dst, uint8_t *src, int size) { memcpy(dst, src, size); }
void TraceParser::move_data(uint8_t *dst, uint8_t *src, int size) { memcpy(dst, src, size); }
inline void coalesced_update(PimMemTraceData *fmtd32, int &coalesced_trace_it, PimMemTraceData *fmtd16, int &trace_it,
                             char prev_cmd, uint64_t &prev_addr)
{
    fmtd32[coalesced_trace_it].cmd = prev_cmd;
    fmtd32[coalesced_trace_it].block_id = fmtd16[trace_it].block_id;
    fmtd32[coalesced_trace_it].thread_id = fmtd16[trace_it].thread_id;
    fmtd32[coalesced_trace_it].addr = fmtd16[trace_it].addr;
    coalesced_trace_it++;
    prev_addr = fmtd16[trace_it].addr & MASK;
}

void TraceParser::coalesce_trace(PimMemTraceData *fmtd32, int *fmtd32_size, PimMemTraceData *fmtd16, int fmtd16_size)
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
                    if ((prev_addr) == (fmtd16[trace_it].addr & MASK)) {
                        if (fmtd16[trace_it].addr < fmtd32[coalesced_trace_it - 1].addr) {
                            fmtd32[coalesced_trace_it - 1].thread_id = fmtd16[trace_it].thread_id;
                            fmtd32[coalesced_trace_it - 1].addr = fmtd16[trace_it].addr & MASK;
                        }
                        VLOG(3) << "Coalescing two read on address 0x" << std::hex << prev_addr << " and 0x"
                                << fmtd16[trace_it].addr << "\n";
                    } else
                        coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'R', prev_addr);
                } else {
                    coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'R', prev_addr);
                }
                prev_cmd = 'R';
                break;
            case 'O':
                if (prev_addr != -1 && prev_cmd == 'O') {
                    if ((prev_addr) == (fmtd16[trace_it].addr & MASK)) {
                        if (fmtd16[trace_it].addr < fmtd32[coalesced_trace_it - 1].addr) {
                            fmtd32[coalesced_trace_it - 1].thread_id = fmtd16[trace_it].thread_id;
                            fmtd32[coalesced_trace_it - 1].addr = fmtd16[trace_it].addr & MASK;
                        }
                        VLOG(3) << "Coalescing two read output on address 0x" << std::hex << prev_addr << " and 0x"
                                << fmtd16[trace_it].addr << "\n";
                    } else
                        coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'O', prev_addr);
                } else {
                    coalesced_update(fmtd32, coalesced_trace_it, fmtd16, trace_it, 'O', prev_addr);
                }
                prev_cmd = 'O';
                break;
            case 'W':
                if (prev_addr != -1 && prev_cmd == 'W') {
                    if ((prev_addr) == (fmtd16[trace_it].addr & MASK)) {
                        // Writes came out of order. Move the data before appending
                        if (fmtd16[trace_it].addr < fmtd32[coalesced_trace_it - 1].addr) {
                            move_data(fmtd32[coalesced_trace_it - 1].data + 16, fmtd32[coalesced_trace_it - 1].data,
                                      16);
                            append_data(fmtd32[coalesced_trace_it - 1].data, fmtd16[trace_it].data, 16);
                            fmtd32[coalesced_trace_it - 1].thread_id = fmtd16[trace_it].thread_id;
                            fmtd32[coalesced_trace_it - 1].addr = fmtd16[trace_it].addr & MASK;
                        } else {
                            append_data(fmtd32[coalesced_trace_it - 1].data + 16, fmtd16[trace_it].data, 16);
                        }
                        VLOG(3) << "Coalescing two writes on address 0x" << std::hex << prev_addr << " and 0x"
                                << fmtd16[trace_it].addr << "\n";
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
} /* namespace pim */
