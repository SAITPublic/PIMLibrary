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
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";
    get_fim_block_info(&fbi_);
}

FimEmulator* FimEmulator::get_instance(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    static FimEmulator* instance_ = new FimEmulator();

    return instance_;
}

int FimEmulator::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " Intialization done ";
    int ret = 0;

    return ret;
}

int FimEmulator::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    return ret;
}

int FimEmulator::convert_mem_trace_from_16B_to_32B(FimMemTraceData* fmtd32, int* fmtd32_size, FimMemTraceData* fmtd16,
                                                   int fmtd16_size, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    TraceParser trace_converter;
    trace_converter.coalesce_trace(fmtd32, fmtd32_size, fmtd16, fmtd16_size);

    DLOG(INFO) << "fmtd16_size : " << fmtd16_size;

#ifdef DEBUG_FIM
    char str[256];
    const char* op_str = get_fim_op_string(op_type);
    sprintf(str, "./test_vectors/dump/%s/fmtd16.dat", op_str);
    dump_fmtd<16>(str, fmtd16, fmtd16_size);
    sprintf(str, "./test_vectors/dump/%s/fmtd32.dat", op_str);
    dump_fmtd<32>(str, fmtd32, fmtd32_size[0]);
#endif

    return ret;
}

int FimEmulator::execute_gemv(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size,
                              FimOpType op_type, uint64_t fim_base_addr, uint8_t* temp_buf)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int num_out_before_reduce = 0;
    uint16_t* sim_output = nullptr;
    int output_size_before_reduce = 0;

    int out_dim = fim_data->bshape.h * output->bshape.n;
    num_out_before_reduce = out_dim * fbi_.num_out_per_grf;
    output_size_before_reduce = num_out_before_reduce * sizeof(uint16_t);
    sim_output = new uint16_t[out_dim];

#ifdef DEBUG_FIM
    fim_sim_.initialize("../external_libs/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "../external_libs/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#else
    fim_sim_.initialize("/opt/rocm/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "/opt/rocm/include/dramsim2/ini/system_hbm_vega20_gemv.ini", 256 * 64 * 2, 64, 1);
#endif

    uint64_t tmp_data_addr = reinterpret_cast<uint64_t>(temp_buf);
    uint64_t fim_data_addr = reinterpret_cast<uint64_t>(fim_data->data);
    uint64_t output_addr = reinterpret_cast<uint64_t>(output->data);

    fim_sim_.alloc_burst(fim_data->size, output_size_before_reduce);
    fim_sim_.preload_data_with_addr(fim_data_addr - fim_base_addr, fim_data->data, fim_data->size);
    fim_sim_.execute_kernel((void*)fmtd32, fmtd32_size);
    fim_sim_.read_result_gemv(tmp_data_addr - fim_base_addr, out_dim);
    fim_sim_.get_reduced_result(sim_output, out_dim);

    if (output->mem_type != MEM_TYPE_HOST) {
        for (int i = 0; i < output->bshape.n; i++) {
            hipMemcpy((half*)output->data + i * fim_data->bshape_r.h, (half*)sim_output + i * fim_data->bshape.h,
                      fim_data->bshape_r.h * sizeof(half), hipMemcpyHostToDevice);
        }
    }

    delete sim_output;

    return ret;
}

int FimEmulator::execute_gemv_add(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size,
                                  FimOpType op_type, uint64_t fim_base_addr, uint8_t* temp_buf)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int num_out_before_reduce = 0;
    uint16_t* sim_output = nullptr;
    int output_size_before_reduce = 0;

    int out_dim = fim_data->bshape.h * output->bshape.n;
    num_out_before_reduce = out_dim * fbi_.num_out_per_grf;
    output_size_before_reduce = num_out_before_reduce * sizeof(uint16_t);
    sim_output = new uint16_t[out_dim];

#ifdef DEBUG_FIM
    fim_sim_.initialize("../external_libs/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "../external_libs/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#else
    fim_sim_.initialize("/opt/rocm/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "/opt/rocm/include/dramsim2/ini/system_hbm_vega20_gemv.ini", 256 * 64 * 2, 64, 1);
#endif

    uint64_t tmp_data_addr = reinterpret_cast<uint64_t>(temp_buf);
    uint64_t fim_data_addr = reinterpret_cast<uint64_t>(fim_data->data);
    uint64_t output_addr = reinterpret_cast<uint64_t>(output->data);

    fim_sim_.alloc_burst(fim_data->size, output_size_before_reduce);
    fim_sim_.preload_data_with_addr(fim_data_addr - fim_base_addr, fim_data->data, fim_data->size);
    fim_sim_.execute_kernel((void*)fmtd32, fmtd32_size);
    fim_sim_.read_result_gemv(tmp_data_addr - fim_base_addr, out_dim);
    fim_sim_.get_reduced_result(sim_output, out_dim);
    fim_sim_.eltwise_add(output->data, sim_output, fim_data->bshape_r.h, fim_data->bshape.h, output->bshape.n);

    delete sim_output;

    return ret;
}

int FimEmulator::execute_bn(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int num_element = 0;
    uint16_t* sim_output = nullptr;
    int sim_output_size = 0;
    int fp16_burst_size = 16;

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
    fim_sim_.execute_kernel_bn((void*)fmtd32, (size_t)fmtd32_size, output->bshape.n, output->bshape.c,
                               output->bshape.w / fp16_burst_size);
    fim_sim_.get_uint16_result(sim_output, num_element);
    if (output->mem_type != MEM_TYPE_HOST)
        hipMemcpy((void*)output->data, (void*)sim_output, output->size, hipMemcpyHostToDevice);

    delete sim_output;

    return ret;
}

int FimEmulator::execute_elt_op(FimBo* output, FimBo* operand0, FimBo* operand1, FimMemTraceData* fmtd32,
                                int fmtd32_size, uint64_t fim_base_addr)
{
    DLOG(INFO) << "called";
    int ret = 0;
    int num_element = 0;
    uint16_t* sim_output = nullptr;
    int sim_output_size = 0;
    int fp16_burst_size = 16;

    num_element = output->size / sizeof(uint16_t);
    sim_output = new uint16_t[num_element];
#ifdef DEBUG_FIM
    fim_sim_.initialize("../external_libs/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "../external_libs/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#else
    fim_sim_.initialize("/opt/rocm/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "/opt/rocm/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#endif

    uint64_t input0_addr = reinterpret_cast<uint64_t>(operand0->data);
    uint64_t input1_addr = reinterpret_cast<uint64_t>(operand1->data);
    uint64_t output_addr = reinterpret_cast<uint64_t>(output->data);

    fim_sim_.alloc_burst(operand0->size, output->size);
    fim_sim_.preload_data_with_addr(input0_addr - fim_base_addr, operand0->data, operand0->size);
    fim_sim_.preload_data_with_addr(input1_addr - fim_base_addr, operand1->data, operand1->size);
    fim_sim_.execute_kernel((void*)fmtd32, (size_t)fmtd32_size);
    fim_sim_.read_result(output_addr - fim_base_addr, output->size);
    fim_sim_.get_uint16_result(sim_output, num_element);
    if (output->mem_type != MEM_TYPE_HOST)
        hipMemcpy((void*)output->data, (void*)sim_output, output->size, hipMemcpyHostToDevice);

    delete sim_output;

    return ret;
}

int FimEmulator::execute_relu(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size,
                              uint64_t fim_base_addr)
{
    DLOG(INFO) << "called";
    int ret = 0;
    int num_element = 0;
    uint16_t* sim_output = nullptr;
    int sim_output_size = 0;
    int fp16_burst_size = 16;

    num_element = output->size / sizeof(uint16_t);
    sim_output = new uint16_t[num_element];
#ifdef DEBUG_FIM
    fim_sim_.initialize("../external_libs/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "../external_libs/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#else
    fim_sim_.initialize("/opt/rocm/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "/opt/rocm/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#endif
    uint64_t fim_data_addr = reinterpret_cast<uint64_t>(fim_data->data);
    uint64_t output_addr = reinterpret_cast<uint64_t>(output->data);

    fim_sim_.alloc_burst(fim_data->size, output->size);
    fim_sim_.preload_data_with_addr(fim_data_addr - fim_base_addr, fim_data->data, fim_data->size);
    fim_sim_.execute_kernel((void*)fmtd32, (size_t)fmtd32_size);
    fim_sim_.read_result(output_addr - fim_base_addr, output->size);
    fim_sim_.get_uint16_result(sim_output, num_element);
    if (output->mem_type != MEM_TYPE_HOST)
        hipMemcpy((void*)output->data, (void*)sim_output, output->size, hipMemcpyHostToDevice);

    delete sim_output;

    return ret;
}

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */
