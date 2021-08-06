#include <iostream>
#include "pim_runtime_api.h"
#include "hip/hip_runtime.h"
#include <torch/script.h>
#include <torch/torch.h>

void KernelLauncher()
{
    std::cout << "Launcher for PIM_Init" << std::endl;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
}

void py_pim_init() { KernelLauncher(); }

static auto registry = torch::RegisterOperators("custom_ops::py_pim_init", &py_pim_init);
