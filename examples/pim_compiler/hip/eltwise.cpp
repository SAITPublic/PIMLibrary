#include "iostream"
#include "pim_runtime_api.h"
#include "pim_compiler.h"

using namespace std;
using namespace pimc;
using namespace pimc::frontend;
using half_float::half;

int pimc_eltwise_add(int h, int w){

    int32_t ret = 0;
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimDesc* pim_desc = PimCreateDesc(1, 1, h, w, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input0 = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_input1 = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST);

    PimBo* pim_input0 = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* pim_input1 = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    // TODO load golden data

    /* __PIM_API__ call : Preload weight data on PIM memory */
    PimCopyMemory(pim_input0, host_input0, HOST_TO_PIM);
    PimCopyMemory(pim_input1, host_input1, HOST_TO_PIM);

    //Declare variables
    IndexVar i(0, h, "i");
    IndexVar j(0, w, "j");
    Buffer B(h, w, "B");
    Buffer C(h, w, "C");
    Var D("D");

    //Define Computation
    D[i][j] = C[i][j] + B[i][j];

    //Run
    PimTarget target = PimCreateTarget(RT_TYPE_HIP, PIM_FP16, GPU);
    PimCompileObj* obj = PimBuildProgram(D, {B, C}, {pim_input0, pim_input1}, target);
    PimBo* device_output = PimExecuteProgram(obj, target);
    PimCopyMemory(host_output, device_output, PIM_TO_HOST);
    PimDeinitialize();

    PimDestroyBo(host_input0);
    PimDestroyBo(host_input1);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input0);
    PimDestroyBo(pim_input1);
    PimDestroyDesc(pim_desc);
    return 0;
}


}

TEST(PimcompilerIntegrationTest, EltAdd) { EXPECT_TRUE(pimc_eltwise_add(256, 1024) == 0); }
