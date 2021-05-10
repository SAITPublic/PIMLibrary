#include <iostream>
#include "pim_runtime_api.h"

#include <torch/script.h>
#include <torch/torch.h>
#include "hip/hip_fp16.h"

void KernelLauncher(const void* inp_data, const int N, void* out_data)
{
    std::cout << "Launcher for PIM_Activation" << std::endl;

    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, N, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* pim_input = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    /* __PIM_API__ call : Copy input data on PIM memory */
    PimCopyMemory(pim_input->data, (void*)inp_data, sizeof(half) * N, HOST_TO_PIM);

    std::cout << "Calling PIMExecuteRelu" << std::endl;
    /* __PIM_API__ call : Execute PIM kernel (Relu) */
    PimExecuteRelu(device_output, pim_input);

    PimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(half) * N, PIM_TO_HOST);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);
    PimDestroyDesc(pim_desc);
}

torch::Tensor py_pim_activation(torch::Tensor input)
{
    const int N = input.numel();  // compute num elements in tensor
    auto inp_data = torch::flatten(input).data_ptr<at::Half>();

    torch::Tensor output = torch::zeros_like(input);
    auto out_data = output.data_ptr<at::Half>();

    KernelLauncher(inp_data, N, out_data);

    return output;
}

static auto registry = torch::RegisterOperators("custom_ops::py_pim_activation", &py_pim_activation);
