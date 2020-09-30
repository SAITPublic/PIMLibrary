#include <iostream>
#include "fim_runtime_api.h"

#include <torch/script.h>
#include <torch/torch.h>
#include "hip/hip_fp16.h"

void KernelLauncher(const void* inp_data, const int N, void* out_data)
{
    std::cout << "Launcher for FIM_Activation" << std::endl;

    FimDesc* fim_desc = FimCreateDesc(1, 1, 1, N, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* fim_input = FimCreateBo(fim_desc, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_FIM);

    /* __FIM_API__ call : Copy input data on FIM memory */
    FimCopyMemory(fim_input->data, (void*)inp_data, sizeof(half) * N, HOST_TO_FIM);

    std::cout << "Calling FIMExecuteRelu" << std::endl;
    /* __FIM_API__ call : Execute FIM kernel (Relu) */
    FimExecuteRelu(device_output, fim_input);

    FimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(half) * N, FIM_TO_HOST);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);
    FimDestroyDesc(fim_desc);
}

torch::Tensor py_fim_activation(torch::Tensor input)
{
    const int N = input.numel();  // compute num elements in tensor
    auto inp_data = torch::flatten(input).data_ptr<at::Half>();

    torch::Tensor output = torch::zeros_like(input);
    auto out_data = output.data_ptr<at::Half>();

    KernelLauncher(inp_data, N, out_data);

    return output;
}

static auto registry = torch::RegisterOperators("custom_ops::py_fim_activation", &py_fim_activation);
