#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "executor/PimCompilerDriver.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_profile.h"
#include "utility/pim_util.h"

namespace pim
{
namespace runtime
{
namespace pimc_driver
{
bool HIPCodegen::execute() { return false; }
void HIPCompiler::execute(std::string hip_kernel, std::string crf_binary)
{
#if PIM_COMPILER_ENABLE == 1
    //std::string hip_kernel = pim_op->get_gpu_kernel();

    if (const char *env_p = std::getenv("SAVE_GENERATED_KERNEL")) {
        int value = *((int *)env_p);
        if (value == 1) {
            std::ofstream hip_file("output_hip_kernel.txt");
            hip_file << hip_kernel;
            hip_file.close();

            std::ofstream crf_file("output_crf_binary.txt");
            crf_file << crf_binary;
            crf_file.close();
        }
    }
    std::string compile_opts = "-O3 --gpu-architecture=gfx906";
    const char *options[] = {compile_opts.c_str()};

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, hip_kernel.c_str(), "generated_pim_source.cu", 0, nullptr, nullptr);
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);

    if (logSize) {
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);

        DLOG(INFO) << log;
    }

    if (compileResult != HIPRTC_SUCCESS)
        std::cout << "Compilation failed." << std::endl;
    else
        std::cout << "Compilation successfull" << std::endl;

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
#endif
}

bool HIPExecutor::execute()
{
/*#if PIM_COMPILER_ENABLE == 1
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
#endif*/
    return false;
}

hipFunction_t PimCDriver::compile_code(std::string kernel, std::string crf_binary)
{
    compile_.execute(kernel, crf_binary);
    return compile_.get_kernel_function();
}

bool PimCDriver::execute_code(KernelArgs *kargs)
{
    executor_.set_kernel_args(kargs);
    return executor_.execute();
}

}  // namespace pimc_driver
}  // namespace runtime
}  // namespace pim
