#include "api/pim_compiler.hpp"
#include "iostream"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"

#include "gtest/gtest.h"

using namespace std;
using namespace pimc;
using namespace pimc::frontend;
using half_float::half;

int pimc_mad(int w)
{
    int32_t ret = 0;
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, w, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input0 = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_input1 = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* pim_input0 = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* pim_input1 = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    // Load input data
    set_rand_half_data((half_float::half*)host_input0->data, (half_float::half)0.5, w);
    set_rand_half_data((half_float::half*)host_input1->data, (half_float::half)0.5, w);

    // Calculate Golden data - compute in CPU
    for (int i = 0; i < w; i++) {
        ((half_float::half*)golden_output->data)[i] =
            5 + (((half_float::half*)host_input0->data)[i] * ((half_float::half*)host_input1->data)[i]);
    }

    PimCopyMemory(pim_input0, host_input0, HOST_TO_PIM);
    PimCopyMemory(pim_input1, host_input1, HOST_TO_PIM);

    // Declare variables
    IndexVar i(0, w, "i");
    ConstVar c(5, "c");
    Buffer vec_x(w, "vec_x");
    Buffer vec_y(w, "vec_y");
    Var D("D");

    // Define Computation
    D[i] = vec_x[i] * vec_y[i] + c;

    // Run
    PimTarget* target = PimCreateTarget(RT_TYPE_HIP, PIM_FP16, GPU);
    PimCompiledObj* obj = PimBuildProgram(D, {vec_x, vec_y}, {pim_input0, pim_input1}, target);
    PimBo* device_output = PimExecuteProgram(obj, target);
    PimCopyMemory(host_output, device_output, PIM_TO_HOST);
    ret = compare_half_relative((half*)host_output->data, (half*)golden_output->data, w);

    //    for (int i = 0; i < 5; i++) {
    //        std::cout<<"PIM Output = " << ((half*)host_output->data)[i] <<" , Golden output = " <<
    //        ((half*)golden_output->data)[i] << std::endl;
    //    }

    PimDestroyBo(host_input0);
    PimDestroyBo(host_input1);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input0);
    PimDestroyBo(pim_input1);
    PimDestroyDesc(pim_desc);
    PimDestroyTarget(target);
    PimDestroyProgram(obj);
    PimDeinitialize();
    return ret;
}

TEST(PimCompilerIntegrationTestMad, Mad) { EXPECT_TRUE(pimc_mad(10) == 0); }
