#include "api/pim_compiler.hpp"
#include "iostream"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"

#include "gtest/gtest.h"

using namespace std;
using namespace pimc;
using namespace pimc::frontend;
using half_float::half;

int pimc_blas_scal(int w)
{
    int32_t ret = 0;
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, w, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input0 = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* pim_input0 = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    // Load input data
    set_rand_half_data((half_float::half*)host_input0->data, (half_float::half)0.5, w);

    //    for (int i = 0; i < w; i++)
    //      ((half_float::half*)host_input0->data)[i] = 2;
    // Calculate Golden data - compute in CPU
    for (int i = 0; i < w; i++) {
        ((half_float::half*)golden_output->data)[i] = 5 * ((half_float::half*)host_input0->data)[i];
    }

    PimCopyMemory(pim_input0, host_input0, HOST_TO_PIM);

    // Declare variables
    IndexVar i(0, w, "i");
    ConstVar a(5, "a");
    Buffer B(w, "B");
    Var D("D");

    // Define Computation
    D[i] = a * B[i];

    // Run
    PimTarget* target = PimCreateTarget(RT_TYPE_HIP, PIM_FP16, GPU);
    PimCompiledObj* obj = PimBuildProgram(D, {B}, {pim_input0}, target);
    PimBo* device_output = PimExecuteProgram(obj, target);
    PimCopyMemory(host_output, device_output, PIM_TO_HOST);
    ret = compare_half_relative((half*)host_output->data, (half*)golden_output->data, w);

    PimDestroyBo(host_input0);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input0);
    PimDestroyDesc(pim_desc);
    PimDestroyTarget(target);
    PimDestroyProgram(obj);
    PimDeinitialize();
    return ret;
}

TEST(PimCompilerIntegrationTestScal, BlasScal) { EXPECT_TRUE(pimc_blas_scal(256) == 0); }
