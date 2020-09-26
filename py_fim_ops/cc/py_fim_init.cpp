#include <iostream>
#include "fim_runtime_api.h"

#include <torch/script.h>
#include <torch/torch.h>

void KernelLauncher()
{
    std::cout << "Launcher for FIM_Init" << std::endl;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);
}

void py_fim_init() { KernelLauncher(); }

static auto registry = torch::RegisterOperators("custom_ops::py_fim_init", &py_fim_init);
