#include <iostream>
#include "fim_runtime_api.h"

#include <torch/script.h>
#include <torch/torch.h>

void KernelLauncher()
{
    std::cout << "Launcher for FIM_Deinit" << std::endl;

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();
}

void py_fim_deinit() { KernelLauncher(); }

static auto registry = torch::RegisterOperators("custom_ops::py_fim_deinit", &py_fim_deinit);
