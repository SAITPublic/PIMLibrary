#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"

#define WIDTH (2048)
#define HEIGHT (1024)
#define INPUT (2048)
#define OUTPUT (1024)

using namespace std;

int main(void)
{
    int i, w, h;
    int errors;

    FimBo hostInput = {.size = INPUT * sizeof(FP16), .memType = MEM_TYPE_HOST};
    FimBo hostWeight = {.size = WIDTH * HEIGHT * sizeof(FP16), .memType = MEM_TYPE_HOST};
    FimBo hostOutput = {.size = OUTPUT * sizeof(FP16), .memType = MEM_TYPE_HOST};
    FimBo goldenOutput = {.size = OUTPUT * sizeof(FP16), .memType = MEM_TYPE_HOST};
    FimBo deviceInput = {.size = INPUT * sizeof(FP16), .memType = MEM_TYPE_DEVICE};
    FimBo deviceWeight = {.size = WIDTH * HEIGHT * sizeof(FP16), .memType = MEM_TYPE_DEVICE};
    FimBo deviceOutput = {.size = OUTPUT * sizeof(FP16), .memType = MEM_TYPE_DEVICE};
    FimBo fimWeight = {.size = WIDTH * HEIGHT * sizeof(FP16), .memType = MEM_TYPE_FIM};

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate host(CPU) memory */
    FimAllocMemory(&hostInput);
    FimAllocMemory(&hostWeight);
    FimAllocMemory(&hostOutput);
    FimAllocMemory(&goldenOutput);
    /* __FIM_API__ call : Allocate device(GPU) memory */
    FimAllocMemory(&deviceInput);
    FimAllocMemory(&deviceOutput);
    /* __FIM_API__ call : Allocate device(FIM) memory */
    FimAllocMemory(&fimWeight);

    /* Initialize the input, weight, output data */
    for (i = 0; i < INPUT; i++) {
        ((FP16*)hostInput.data)[i] = convertF2H(2.0f);
    }
    for (h = 0; h < HEIGHT; h++) {
        for (w = 0; w < WIDTH; w++) {
            ((FP16*)hostWeight.data)[h * WIDTH + w] = convertF2H(1.0f);
        }
    }
    for (i = 0; i < OUTPUT; i++) {
        ((FP16*)hostOutput.data)[i] = convertF2H(0.0f);
        ((FP16*)goldenOutput.data)[i] = convertF2H(0.0f);
    }

    /* __FIM_API__ call : Copy operand data from host to device */
    FimCopyMemory(&deviceInput, &hostInput, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(&fimWeight, &hostWeight, OP_GEMV);

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecute(&deviceOutput, &deviceInput, &fimWeight, OP_GEMV);

    /* __FIM_API__ call : Copy output data from device to host */
    FimCopyMemory(&hostOutput, &deviceOutput, FIM_TO_HOST);

    /* Verify the results */
    errors = 0;
    FP16* weightPtr = (FP16*)hostWeight.data;
    float tempOutput;
    for (h = 0; h < HEIGHT; h++) {
        tempOutput = 0.0f;
        for (w = 0; w < WIDTH; w++) {
            tempOutput += convertH2F(((FP16*)hostInput.data)[w]) * convertH2F(((FP16*)weightPtr)[w]);
        }
        ((FP16*)goldenOutput.data)[h] = convertF2H(tempOutput);
        if (((FP16*)goldenOutput.data)[h] != ((FP16*)hostOutput.data)[h]) errors++;
        weightPtr += WIDTH;
    }
    if (errors != 0)
        printf("\t\t\t\t\tFAILED: %d errors\n", errors);
    else
        printf("\t\t\t\t\tPASSED!\n");

    /* __FIM_API__ call : Free device(GPU) memory */
    FimFreeMemory(&deviceInput);
    /* __FIM_API__ call : Free device(FIM) memory */
    FimFreeMemory(&fimWeight);
    /* __FIM_API__ call : Free host(CPU) memory */
    FimFreeMemory(&hostInput);
    FimFreeMemory(&hostWeight);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return errors;
}
