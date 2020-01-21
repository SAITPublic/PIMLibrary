#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_fp16.h"
#include "fim_runtime_api.h"

#define LENGTH	(1024*1024)

using namespace std;

int main(void)
{
    int i;
    int errors;
    _Float16* hostInput0 = nullptr;
    _Float16* hostInput1 = nullptr;
    _Float16* hostOutput = nullptr;
    _Float16* deviceInput0 = nullptr;
    _Float16* deviceInput1 = nullptr;
    _Float16* deviceOutput = nullptr;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate host(CPU) memory */
    FimAllocMemory((void**)&hostInput0, LENGTH * sizeof(_Float16), MEM_TYPE_HOST);
    FimAllocMemory((void**)&hostInput1, LENGTH * sizeof(_Float16), MEM_TYPE_HOST);
    FimAllocMemory((void**)&hostOutput, LENGTH * sizeof(_Float16), MEM_TYPE_HOST);

    /* Initialize the input data */
    for (i = 0; i < LENGTH; i++) {
        hostInput0[i] = (_Float16)0.0;
        hostInput1[i] = (_Float16)i;
    }

    /* __FIM_API__ call : Allocate device(GPU) memory */
    FimAllocMemory((void**)&deviceInput0, LENGTH * sizeof(_Float16), MEM_TYPE_DEVICE);
    FimAllocMemory((void**)&deviceInput1, LENGTH * sizeof(_Float16), MEM_TYPE_DEVICE);
    FimAllocMemory((void**)&deviceOutput, LENGTH * sizeof(_Float16), MEM_TYPE_DEVICE);

    /* __FIM_API__ call : Replacement data */
    FimDataReplacement(deviceInput1, LENGTH * sizeof(_Float16), OP_ELT_ADD);

    /* __FIM_API__ call : Copy operand data from host to device */
    FimCopyMemory(deviceInput0, hostInput0, LENGTH * sizeof(_Float16), HOST_TO_DEVICE);
    FimCopyMemory(deviceInput1, hostInput1, LENGTH * sizeof(_Float16), HOST_TO_FIM);

    /* __FIM_API__ call : Execute FIM kernel (Element-wise Add) */
    FimExecute(deviceOutput, deviceInput0, deviceInput1, LENGTH, OP_ELT_ADD);

    /* __FIM_API__ call : Copy output data from device to host */
    FimCopyMemory(hostOutput, deviceOutput, LENGTH * sizeof(_Float16), FIM_TO_HOST);

    /* Verify the results */
    errors = 0;
    for (i = 0; i < LENGTH; i++) {
        if (hostInput0[i] + hostInput1[i] != hostOutput[i])
            errors++;
    }
    if (errors != 0)
        printf("\t\t\t\t\tFAILED: %d errors\n",errors);
    else
        printf("\t\t\t\t\tPASSED!\n");

    /* __FIM_API__ call : Free device(GPU) memory */
    FimFreeMemory(deviceInput0, MEM_TYPE_DEVICE);
    FimFreeMemory(deviceInput1, MEM_TYPE_FIM);

    /* __FIM_API__ call : Free host(CPU) memory */
    FimFreeMemory(hostInput0, MEM_TYPE_HOST);
    FimFreeMemory(hostInput1, MEM_TYPE_HOST);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return errors;
}
