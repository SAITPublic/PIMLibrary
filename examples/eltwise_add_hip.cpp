#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "fim_runtime_api.h"

#define LENGTH	(1024*1024)

using namespace std;

int main(void)
{
    int i;
    int errors;
    float* hostInput0 = nullptr;
    float* hostInput1 = nullptr;
    float* hostOutput = nullptr;
    float* deviceInput0 = nullptr;
    float* deviceInput1 = nullptr;
    float* deviceOutput = nullptr;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize();

    /* __FIM_API__ call : Allocate host(CPU) memory */
    FimAllocMemory((float**)&hostInput0, LENGTH * sizeof(float), MEM_TYPE_HOST);
    FimAllocMemory((float**)&hostInput1, LENGTH * sizeof(float), MEM_TYPE_HOST);
    FimAllocMemory((float**)&hostOutput, LENGTH * sizeof(float), MEM_TYPE_HOST);

    /* Initialize the input data */
    for (i = 0; i < LENGTH; i++) {
        hostInput0[i] = (float)0.0;
        hostInput1[i] = (float)i;
    }

    /* __FIM_API__ call : Allocate device(GPU) memory */
    FimAllocMemory((float**)&deviceInput0, LENGTH * sizeof(float), MEM_TYPE_DEVICE);
    FimAllocMemory((float**)&deviceInput1, LENGTH * sizeof(float), MEM_TYPE_DEVICE);
    FimAllocMemory((float**)&deviceOutput, LENGTH * sizeof(float), MEM_TYPE_DEVICE);

    /* __FIM_API__ call : Copy operand data from host to device */
    FimCopyMemory(deviceInput0, hostInput0, LENGTH * sizeof(float), HOST_TO_FIM);
    FimCopyMemory(deviceInput1, hostInput1, LENGTH * sizeof(float), HOST_TO_FIM);

    /* __FIM_API__ call : Execute FIM kernel (Element-wise Add) */
    FimExecute(deviceOutput, deviceInput0, deviceInput1, LENGTH, OP_ELT_ADD, FIM_FP16);

    /* __FIM_API__ call : Copy output data from device to host */
    FimCopyMemory(hostOutput, deviceOutput, LENGTH * sizeof(float), FIM_TO_HOST);

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
    FimFreeMemory(deviceInput1, MEM_TYPE_DEVICE);

    /* __FIM_API__ call : Free host(CPU) memory */
    FimFreeMemory(hostInput0, MEM_TYPE_HOST);
    FimFreeMemory(hostInput1, MEM_TYPE_HOST);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return errors;
}
