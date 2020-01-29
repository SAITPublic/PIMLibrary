#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"

#define LENGTH (1024 * 1024)

using namespace std;

int main(void)
{
    int i;
    int errors;
    _Float16* hostInput = nullptr;
    _Float16* hostWeight = nullptr;
    _Float16* hostOutput = nullptr;
    _Float16* deviceInput = nullptr;
    _Float16* deviceOutput = nullptr;
    _Float16* fimWeight = nullptr;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate host(CPU) memory */
    FimAllocMemory((void**)&hostInput, LENGTH * sizeof(_Float16), MEM_TYPE_HOST);
    FimAllocMemory((void**)&hostWeight, LENGTH * sizeof(_Float16), MEM_TYPE_HOST);
    FimAllocMemory((void**)&hostOutput, LENGTH * sizeof(_Float16), MEM_TYPE_HOST);
    /* __FIM_API__ call : Allocate device(GPU) memory */
    FimAllocMemory((void**)&deviceInput, LENGTH * sizeof(_Float16), MEM_TYPE_DEVICE);
    FimAllocMemory((void**)&deviceOutput, LENGTH * sizeof(_Float16), MEM_TYPE_DEVICE);
    /* __FIM_API__ call : Allocate device(FIM) memory */
    FimAllocMemory((void**)&fimWeight, LENGTH * sizeof(_Float16), MEM_TYPE_FIM);

    /* Initialize the input data */
    for (i = 0; i < LENGTH; i++) {
        hostInput[i] = (_Float16)0.0;
        hostWeight[i] = (_Float16)i;
    }

    /* __FIM_API__ call : Copy operand data from host to device */
    FimCopyMemory(deviceInput, hostInput, LENGTH * sizeof(_Float16), HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(fimWeight, hostWeight, LENGTH * sizeof(_Float16), OP_ELT_ADD);

    /* __FIM_API__ call : Execute FIM kernel (Element-wise Add) */
    FimExecute(deviceOutput, deviceInput, fimWeight, LENGTH, OP_ELT_ADD);

    /* __FIM_API__ call : Copy output data from device to host */
    FimCopyMemory(hostOutput, deviceOutput, LENGTH * sizeof(_Float16), FIM_TO_HOST);

    /* Verify the results */
    errors = 0;
    for (i = 0; i < LENGTH; i++) {
        if (hostInput[i] + hostWeight[i] != hostOutput[i]) errors++;
    }
    if (errors != 0)
        printf("\t\t\t\t\tFAILED: %d errors\n", errors);
    else
        printf("\t\t\t\t\tPASSED!\n");

    /* __FIM_API__ call : Free device(GPU) memory */
    FimFreeMemory(deviceInput, MEM_TYPE_DEVICE);
    /* __FIM_API__ call : Free device(FIM) memory */
    FimFreeMemory(fimWeight, MEM_TYPE_FIM);
    /* __FIM_API__ call : Free host(CPU) memory */
    FimFreeMemory(hostInput, MEM_TYPE_HOST);
    FimFreeMemory(hostWeight, MEM_TYPE_HOST);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return errors;
}
