#include <iostream>
#include "pim_runtime_api.h"
#include "hip/hip_runtime.h"
#include <torch/script.h>
#include <torch/torch.h>

void KernelLauncher()
{
    std::cout << "Launcher for PIM_Deinit" << std::endl;

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();
}

void py_pim_deinit() { KernelLauncher(); }

static auto registry = torch::RegisterOperators("custom_ops::py_pim_deinit", &py_pim_deinit);
