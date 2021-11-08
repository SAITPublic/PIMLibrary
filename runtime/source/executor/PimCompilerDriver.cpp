#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "executor/PimCompilerDriver.h"
#include "utility/pim_dump.hpp"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

namespace pim
{
namespace runtime
{
namespace pimc_driver
{
std::string HIPCodegen::get_pim_program(PimOpType op)
{
    std::string pim_pnemonic_program;

    pim_pnemonic_program = "PARK_IN\n SB_TO_HAB\n PROGRAM_CRF\n";
    if (op == OP_ELT_ADD)
        pim_pnemonic_program += "HAB_TO_HABPIM\n ADD_VECTOR\n HABPIM_TO_HAB\n";
    else if (op == OP_ELT_MUL)
        pim_pnemonic_program += "HAB_TO_HABPIM\n MUL_VECTOR\n HABPIM_TO_HAB\n";
    else if (op == OP_RELU)
        pim_pnemonic_program += "HAB_TO_HABPIM\n RELU\n HABPIM_TO_HAB\n";
    else if (op == OP_GEMV)
        pim_pnemonic_program += "GEMV\n";
    else {
        DLOG(INFO) << "Invalid operation" << __FUNCTION__;
        return "";
    }
    pim_pnemonic_program += "HAB_TO_SB\n PARK_OUT";
    return pim_pnemonic_program;
}

bool HIPCodegen::execute(PimOpType op, pimc::PimDeviceConfig device_config)
{
#ifdef RADEON7
    auto target = pimc::CreatePimTarget(pimc::Runtime::PIMC_HIP, device_config, pimc::DeviceAttr::AMD_RADEON7);
#else
    auto target = pimc::CreatePimTarget(pimc::Runtime::PIMC_HIP, device_config, pimc::DeviceAttr::AMD_MI50);
#endif
    pimc::OpDesc op_desc(input_list_, output_list_);

    // Define pim program for elementwise
    std::string program = get_pim_program(op);

    // Generate hip kernel and corresponding pim kernel
    pim_op_ = pimc::BuildPimProgramFromSource(target, op_desc, program);

    pimc::DeletePimTarget(target);
    return true;
}

void HIPCompiler::execute(pimc::PimCCompiled *pim_op)
{
    std::string hip_kernel = pim_op->get_gpu_kernel();

    if (const char *env_p = std::getenv("SAVE_GENERATED_KERNEL")) {
        int value = *((int *)env_p);
        if (value == 1) {
            std::ofstream hip_file("output_hip_kernel.txt");
            hip_file << hip_kernel;
            hip_file.close();

            std::ofstream crf_file("output_crf_binary.txt");
            crf_file << pim_op->get_crf_binary();
            crf_file.close();
        }
    }
    std::string compile_opts = "-O3 --gpu-architecture=gfx906";
    const char *options[] = {compile_opts.c_str()};

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, hip_kernel.c_str(), "genereated_pim_source.cu", 0, nullptr, nullptr);
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);

    if (logSize) {
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);

        DLOG(INFO) << log;
    }

    if (compileResult != HIPRTC_SUCCESS)
        DLOG(INFO) << "Compilation failed.";
    else
        DLOG(INFO) << "Compilation successfull";

    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);

    std::vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());

    hipError_t err;
    hiprtcDestroyProgram(&prog);
    err = hipModuleLoadData(&module_, code.data());
    if (err != hipSuccess) {
        DLOG(INFO) << "Falied to Load Module Err: " << err << std::endl;
    }
    err = hipModuleGetFunction(&kernel_, module_, "pim_hip_kernel");
    if (err != hipSuccess) {
        DLOG(INFO) << "Falied to Load function Err: " << err << std::endl;
    }
}

bool HIPExecutor::execute(PimOpType op_type)
{
    unsigned blocks = 64;
    switch (op_type) {
        case OP_ELT_ADD:
        case OP_ELT_MUL: {
            unsigned threads_per_block = 32;
            EltArgs<half_float::half> *elt_args = reinterpret_cast<EltArgs<half_float::half> *>(kargs_);
            hipModuleLaunchKernel(elt_args->get_kernel(), blocks, 1, 1, threads_per_block, 1, 1, 0, nullptr, NULL,
                                  elt_args->get_kconfig());
        } break;
        case OP_GEMV: {
            unsigned threads_per_block = 64;
            GemvKArgs<half_float::half> *gemv_args = reinterpret_cast<GemvKArgs<half_float::half> *>(kargs_);
            hipModuleLaunchKernel(gemv_args->get_kernel(), blocks, 1, 1, threads_per_block, 1, 1, 0, nullptr, NULL,
                                  gemv_args->get_kconfig());
        } break;
        case OP_RELU: {
            unsigned threads_per_block = 32;
            ReluArgs<half_float::half> *relu_args = reinterpret_cast<ReluArgs<half_float::half> *>(kargs_);
            hipModuleLaunchKernel(relu_args->get_kernel(), blocks, 1, 1, threads_per_block, 1, 1, 0, nullptr, NULL,
                                  relu_args->get_kconfig());
        }; break;
        default:
            DLOG(INFO) << "Invalid operator type " << __FUNCTION__;
            return false;
    }
    hipStreamSynchronize(nullptr);
    return true;
}

pimc::PimCCompiled *PimCDriver::generate_code(PimOpType op, std::vector<pimc::TensorDesc> input_list,
                                              std::vector<pimc::TensorDesc> output_list)
{
    auto device_config = create_device_config();
    curr_op_type_ = op;

    codegen_.set_input_desc(input_list);
    codegen_.set_output_desc(output_list);
    codegen_.execute(op, device_config);

    return codegen_.get_pim_op();
}

hipFunction_t PimCDriver::compile_code()
{
    compile_.execute(codegen_.get_pim_op());
    return compile_.get_kernel_function();
}

bool PimCDriver::execute_code(KernelArgs *kargs)
{
    executor_.set_pim_op(codegen_.get_pim_op());
    executor_.set_kernel_args(kargs);
    return executor_.execute(curr_op_type_);
}
}
}
}
