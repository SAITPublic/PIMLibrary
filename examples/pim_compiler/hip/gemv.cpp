#include "iostream"
#include "pim_runtime_api.h"
#include "api/pim_compiler.hpp"
#include "utility/pim_debug.hpp"

#include "gtest/gtest.h"

using namespace std;
using namespace pimc;
using namespace pimc::frontend;
using half_float::half;

int pimc_gemv(int h, int w){
    int32_t ret = 0;
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimDesc* pim_desc_in = PimCreateDesc(1, 1, 1, h, PIM_FP16);
    PimDesc* pim_desc_w = PimCreateDesc(1, 1, h, w, PIM_FP16);
    PimDesc* pim_desc_out = PimCreateDesc(1, 1, 1, w, PIM_FP16);

    // __PIM_API__ call : Create PIM Buffer Object
    PimBo* host_input = PimCreateBo(pim_desc_in, MEM_TYPE_HOST);
    PimBo* host_weight = PimCreateBo(pim_desc_w, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc_out, MEM_TYPE_HOST);

    PimBo* pim_input = PimCreateBo(pim_desc_in, MEM_TYPE_PIM);
    PimBo* pim_weight = PimCreateBo(pim_desc_w, MEM_TYPE_PIM);
    PimBo* golden_output = PimCreateBo(pim_desc_out, MEM_TYPE_HOST);
    // Load operand data
    //set_rand_half_data((half_float::half*)host_input0->data, (half_float::half)0.5, (h * w));
    //set_rand_half_data((half_float::half*)host_input1->data, (half_float::half)0.5, w);
    //
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
           ((half_float::half*)host_weight->data)[(i * w) + j] = 0.2;
           ((half_float::half*)host_input->data)[i] = 0.3;
        }
    }

    //Compute Golden output
    for (int j = 0; j < w; j++) {
        ((half_float::half*)golden_output->data)[j] = 0.0f;
        for (int i = 0; i < h; i++)
            ((half_float::half*)golden_output->data)[j] += ((half_float::half*)host_weight->data)[(j * h) + i] * ((half_float::half*)host_input->data)[i];
    }

    //Copy input data from HOST to PIM
    PimCopyMemory(pim_input, host_input, HOST_TO_PIM);
    PimCopyMemory(pim_weight, host_weight, HOST_TO_PIM);

    //Declare variables
    IndexVar i(0, h, "i");
    IndexVar j(0, w, "j");
    Buffer weight(w, h, "weight");
    Buffer input(h, "input");
    Var D("D");

    //Define Computation
    D[j] += weight[i][j] * input[i];

    //Run
    PimTarget* target = PimCreateTarget(RT_TYPE_HIP, PIM_FP16, GPU);
    //Reorder weights for GEMV

    PimCompiledObj* obj = PimBuildProgram(D, {weight, input}, {pim_weight, pim_input}, target);
    PimBo* device_output = PimExecuteProgram(obj, target);
    PimCopyMemory(host_output, device_output, PIM_TO_HOST);
    ret = compare_half_relative((half*)host_output->data, (half*)golden_output->data, w);

    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);
    PimDestroyBo(pim_weight);
    PimDestroyDesc(pim_desc_in);
    PimDestroyDesc(pim_desc_w);
    PimDestroyDesc(pim_desc_out);
    PimDestroyTarget(target);
    PimDestroyProgram(obj);
    PimDeinitialize();
    return ret;
}


TEST(PimCompilerIntegrationTestGemv, GemvMac) { EXPECT_TRUE(pimc_gemv(256, 4096) == 0); }
