#include "utility/fim_util.h"
#include <stdio.h>
#include <stdlib.h>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"

int initTestVector(FimBo& hostInput, FimBo& hostWeight, FimBo& hostOutput)
{
    int ret = 0;
    int i, w, h;
    const int WIDTH = hostInput.size / sizeof(FP16);
    const int INPUT = WIDTH;
    const int HEIGHT = hostWeight.size / hostInput.size / sizeof(FP16);
    const int OUTPUT = hostOutput.size / sizeof(FP16);
    FP16* inputPtr = (FP16*)hostInput.data;
    FP16* weightPtr = (FP16*)hostWeight.data;
    FP16* outputPtr = (FP16*)hostOutput.data;

    for (i = 0; i < INPUT; i++) {
        inputPtr[i] = 1.1234456f;
    }
    for (h = 0; h < HEIGHT; h++) {
        for (w = 0; w < WIDTH; w++) {
            weightPtr[h * WIDTH + w] = 2.98776f;
        }
    }
    for (i = 0; i < OUTPUT; i++) {
        outputPtr[i] = 0.0f;
    }

    return ret;
}

int verifyResult(FimBo& hostInput, FimBo& hostWeight, FimBo& hostOutput, FimBo& deviceOutput)
{
    int ret = 0;
    int errors = 0;
    int i, w, h;
    FP16* inputPtr = (FP16*)hostInput.data;
    FP16* weightPtr = (FP16*)hostWeight.data;
    FP16* outputPtr = (FP16*)hostOutput.data;
    FP16 t_output;
    const int WIDTH = hostInput.size / sizeof(FP16);
    const int HEIGHT = hostWeight.size / hostInput.size / sizeof(FP16);
    const int OUTPUT = hostOutput.size / sizeof(FP16);

    /* __FIM_API__ call : Copy output data from device to host */
    FimCopyMemory(&hostOutput, &deviceOutput, FIM_TO_HOST);

    for (h = 0; h < HEIGHT; h++) {
        t_output = 0.0f;
        for (w = 0; w < WIDTH; w++) {
            t_output += inputPtr[w] * weightPtr[w];
        }
        if (t_output != outputPtr[h]) {
            printf("%f : %f\n", convertH2F(t_output), convertH2F(outputPtr[h]));
            errors++;
            ret = -1;
        }
        weightPtr += WIDTH;
    }
    if (errors != 0)
        printf("\t\t\t\t\tFAILED: %d errors\n", errors);
    else
        printf("\t\t\t\t\tPASSED!\n");

    for (i = 0; i < OUTPUT; i++) {
        outputPtr[i] = 0.0f;
    }
    FimCopyMemory(&deviceOutput, &hostOutput, HOST_TO_FIM);

    return ret;
}
