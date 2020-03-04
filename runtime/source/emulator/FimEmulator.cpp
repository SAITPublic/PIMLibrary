#include "emulator/FimEmulator.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "utility/fim_dump.hpp"
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
namespace emulator
{
FimEmulator::FimEmulator(void) { DLOG(INFO) << "called "; }
FimEmulator* FimEmulator::get_instance(void)
{
    DLOG(INFO) << "Called";
    static FimEmulator* instance_ = new FimEmulator();

    return instance_;
}

int FimEmulator::initialize(void)
{
    DLOG(INFO) << "Intialization done ";
    int ret = 0;

    return ret;
}

int FimEmulator::deinitialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

int FimEmulator::convert_mem_trace_from_16B_to_32B(FimMemTraceData* fmtd32, int* fmtd32_size, FimMemTraceData* fmtd16,
                                                   int fmtd16_size)
{
    DLOG(INFO) << "called";
    int ret = 0;
    char str[256];

    sprintf(str, "../test_vectors/dump/fmtd16_1cu_1th.txt");
    dump_fmtd<16>(str, fmtd16, fmtd16_size);

    TraceParser trace_converter;
    trace_converter.coalesce_trace(fmtd32, fmtd32_size, fmtd16, fmtd16_size);

    sprintf(str, "../test_vectors/dump/fmtd32_1cu_1th.txt");
    dump_fmtd<32>(str, fmtd32, fmtd32_size[0]);

    return ret;
}

int FimEmulator::execute_fim(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size,
                             FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */
