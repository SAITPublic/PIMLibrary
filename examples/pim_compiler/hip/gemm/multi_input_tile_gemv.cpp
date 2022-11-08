#include "api/pim_compiler.hpp"
#include "iostream"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"

#include "gtest/gtest.h"

using namespace std;
using namespace pimc;
using namespace pimc::frontend;
using half_float::half;

int pimc_gemv(int h, int w)
{
    int32_t ret = 0;
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimGemmDesc* gemm_desc = PimCreateGemmDesc(1, 1, 1, h, 1, w, PIM_FP16, I_X_W);
    // PimGemmDesc* gemm_desc = PimCreateGemmDesc(1, 1, h, 1, w, 1, PIM_FP16, W_X_I);

    // __PIM_API__ call : Create PIM Buffer Object
    PimBo* host_input = PimCreateBo(gemm_desc, MEM_TYPE_HOST, GEMM_INPUT);
    PimBo* host_weight = PimCreateBo(gemm_desc, MEM_TYPE_HOST, GEMM_WEIGHT);
    PimBo* host_output = PimCreateBo(gemm_desc, MEM_TYPE_HOST, GEMM_OUTPUT);
    PimBo* golden_output = PimCreateBo(gemm_desc, MEM_TYPE_HOST, GEMM_OUTPUT);

    PimBo* device_input = PimCreateBo(gemm_desc, MEM_TYPE_DEVICE, GEMM_INPUT);
    PimBo* device_weight = PimCreateBo(gemm_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT);
    // Load operand data
    set_rand_half_data((half_float::half*)host_weight->data, (half)0.2, (h * w));
    set_rand_half_data((half_float::half*)host_input->data, (half)0.2, h);

    // Compute Golden output
    for (int j = 0; j < w; j++) {
        ((half*)golden_output->data)[j] = 0.0f;
        for (int i = 0; i < h; i++)
            ((half*)golden_output->data)[j] += ((half*)host_weight->data)[(i * w) + j] * ((half*)host_input->data)[i];
    }

    // Copy input data from HOST to PIM
    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    // Declare variables
    IndexVar i(0, h, "i");
    IndexVar j(0, w, "j");
    Buffer weight(h, w, "weight");
    Buffer input(h, "input");
    Var D("D");

    // Define Computation
    D[j] += weight[j][i] * input[i];

    // Run
    PimTarget* target = PimCreateTarget(RT_TYPE_HIP, PIM_FP16, GPU);
    PimCompiledObj* obj = PimBuildProgram(D, {weight, input}, {device_weight, device_input}, target);
    PimBo* device_output = PimExecuteProgram(obj, target);
    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    ret = compare_half_relative((half*)host_output->data, (half*)golden_output->data, w, 0.01f);

    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyGemmDesc(gemm_desc);
    PimDestroyTarget(target);
    PimDestroyProgram(obj);
    PimDeinitialize();
    return ret;
}

TEST(PimCompilerIntegrationTestGemv, GemvMacMultiInput) { EXPECT_TRUE(pimc_gemv(512, 4096) == 0); }
