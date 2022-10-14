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
        DLOG(INFO) << "Compilation failed." << std::endl;
    else
        DLOG(INFO) << "Compilation successful" << std::endl;

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

bool HIPExecutor::execute() { return false; }
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
