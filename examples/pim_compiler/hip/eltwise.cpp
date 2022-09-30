#include "iostream"
#include "pim_runtime_api.h"
#include "pim_compiler.h"

#define MAT_ROW (256)
#define MAT_COL (1024)

using namespace std;
using namespace pimc;
using namespace pimc::frontend;
using half_float::half;

int pimc_eltwise_add(){

    int32_t ret = 0;
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, input_len, PIM_FP16);

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
    IndexVar i(0, input_len / sizeof(half), "i");
    Buffer B(input_len / sizeof(half), "B");
    Buffer C(input_len / sizeof(half), "C");
    Var D("D");

    //Define Computation
    D[i][j] = C[i][j] + B[i][j];

    //Run
    PimTarget target = PimCreateTarget(RT_TYPE_HIP, PIM_FP16, GPU);
    PimCompileObj* obj = PimBuildProgram(D, {B, C}, {host_input0, host_input1}, target);
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

TEST(PimcompilerIntegrationTest, EltAdd) { EXPECT_TRUE(pimc_eltwise_add(8192) == 0); }
