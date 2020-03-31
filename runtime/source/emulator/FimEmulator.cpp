#include "emulator/FimEmulator.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "utility/fim_dump.hpp"
#include "utility/fim_log.h"
#include "utility/fim_util.h"

namespace fim
{
namespace runtime
{
namespace emulator
{
FimEmulator::FimEmulator(void)
{
    DLOG(INFO) << "called ";
    get_fim_block_info(&fbi_);
}

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
                                                   int fmtd16_size, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    TraceParser trace_converter;
    trace_converter.coalesce_trace(fmtd32, fmtd32_size, fmtd16, fmtd16_size);

#ifdef DEBUG_FIM
    char str[256];
    const char* op_str = get_fim_op_string(op_type);
    sprintf(str, "../test_vectors/dump/%s/fmtd16_1cu_2th.dat", op_str);
    dump_fmtd<16>(str, fmtd16, fmtd16_size);
    sprintf(str, "../test_vectors/dump/%s/fmtd32_1cu_2th.dat", op_str);
    dump_fmtd<32>(str, fmtd32, fmtd32_size[0]);
#endif

    return ret;
}

int FimEmulator::execute_fim(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size,
                             FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;
    int num_element = 0;
    uint16_t* sim_output = nullptr;
    int sim_output_size = 0;

    if (op_type == OP_ELT_ADD) {
        num_element = output->size / sizeof(uint16_t);
        sim_output = new uint16_t[num_element];

#ifdef DEBUG_FIM
        fim_sim_.initialize("../external_libs/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                            "../external_libs/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#else
        fim_sim_.initialize("/opt/rocm/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                            "/opt/rocm/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#endif
        fim_sim_.alloc_burst(fim_data->size, fim_data->size);
        fim_sim_.preload_data(fim_data->data, fim_data->size);
        fim_sim_.execute_kernel((void*)fmtd32, (size_t)fmtd32_size);
        fim_sim_.get_uint16_result(sim_output, num_element);
        if (output->mem_type != MEM_TYPE_HOST)
            hipMemcpy((void*)output->data, (void*)sim_output, output->size, hipMemcpyHostToDevice);
    } else if (op_type == OP_GEMV) {
        num_element = output->size * fbi_.num_out_per_grf / sizeof(uint16_t);
        sim_output = new uint16_t[num_element];
        sim_output_size = num_element * sizeof(uint16_t);
#ifdef DEBUG_FIM
        fim_sim_.initialize("../external_libs/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                            "../external_libs/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#else
        fim_sim_.initialize("/opt/rocm/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                            "/opt/rocm/include/dramsim2/ini/system_hbm_vega20_gemv.ini", 256 * 64 * 2, 64, 1);
#endif
        fim_sim_.alloc_burst(fim_data->size, sim_output_size);
        fim_sim_.preload_data((void*)fim_data->data, fim_data->size);
        fim_sim_.execute_kernel((void*)fmtd32, fmtd32_size);
        fim_sim_.get_uint16_result(sim_output, num_element);
        reduce_sum_for_gemv((void*)sim_output /* out */, (void*)sim_output /* in */, sim_output_size,
                            fbi_.num_out_per_grf);
        if (output->mem_type != MEM_TYPE_HOST)
            hipMemcpy((void*)output->data, (void*)sim_output, output->size, hipMemcpyHostToDevice);
    }

    delete sim_output;

    return ret;
}

int FimEmulator::compare_output(uint16_t* test_output, FimBo* output)
{
    uint16_t* golden_output = static_cast<uint16_t*>(output->data);
    int num_element = output->size / sizeof(uint16_t);
    int num_error = 0;
    int num_warning = 0;
    int num_correct = 0;

    for (int i = 0; i < num_element; i++) {
        uint16_t m = test_output[i];
        uint16_t n = golden_output[i];

        if (uint16_equal(m, n, 4)) {
            num_correct++;
            // std::cout << "correct : "  << m << " and "   << n << " are same " << std::endl;
        } else if (uint16_equal(m, n, 256)) {
            num_warning++;
            // std::cerr << "Warning: " <<" (" << m << " and " << n << ") are similiar "  << std::endl;
        } else {
            num_error++;
            // std::cout << "error : "  << m << " and "   << n << " are not same " << std::endl;
            // return 0;
        }
    }

    std::cout << "error count : " << num_error << std::endl;
    std::cout << "num_warning : " << num_warning << std::endl;
    std::cout << "num correct : " << num_correct << std::endl;
}

bool FimEmulator::uint16_equal(uint16_t A, uint16_t B, int maxUlpsDiff)
{
    if ((A & (1 << 15)) != (B & (1 << 15))) {
        if (A == B) return true;
        return false;
    }

    // Find the difference in ULPs.
    int ulpsDiff = abs(A - B);
    if (ulpsDiff <= maxUlpsDiff) return true;

    return false;
}

int FimEmulator::set_input_for_test(uint16_t* test_input, int test_size)
{
    NumpyBurstType input0_npbst;
    NumpyBurstType input1_npbst;

    input0_npbst.load_fp16("../test_vectors/data/resadd_input0_65536.npy");
    input1_npbst.load_fp16("../test_vectors/data/resadd_input1_65536.npy");

    fim_sim_.set_data_for_test(&input0_npbst, &input1_npbst, test_input);
    fim_sim_.alloc_burst(test_size, test_size);
    fim_sim_.preload_data((void*)test_input, test_size);
}

int FimEmulator::set_kernel_trace_for_test(FimMemTraceData* fmtd32, int* num_trace)
{
    vector<MemTraceData> vec_trace_data;
    MemTraceData* test_trace_data;

    fim_sim_.read_memory_trace("../test_vectors/mem_trace/mem_trace_64ch_1tile.txt", vec_trace_data);
    *num_trace = vec_trace_data.size();
    test_trace_data = new MemTraceData[*num_trace];
    fim_sim_.vector_to_arr(vec_trace_data, test_trace_data);

    for (int i = 0; i < *num_trace; i++) {
        fmtd32[i].addr = test_trace_data[i].addr;
        fmtd32[i].block_id = test_trace_data[i].block_id;
        fmtd32[i].thread_id = test_trace_data[i].thread_id;
        fmtd32[i].cmd = test_trace_data[i].cmd;
        cout << hex << " " << fmtd32[i].addr;
        cout << dec << " b : " << fmtd32[i].block_id << " t : " << fmtd32[i].thread_id << " c : " << fmtd32[i].cmd
             << endl;
        memcpy(fmtd32[i].data, test_trace_data[i].data, sizeof(uint16_t) * 16);
    }

    delete[] test_trace_data;
}

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */
