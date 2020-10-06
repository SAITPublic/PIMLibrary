#include "fim_runtime_api.h"

#include <torch/script.h>
#include <torch/torch.h>
#include "hip/hip_fp16.h"

void KernelLauncher(const void* inp_data, const int N, const int DIMS, const void* mean, const void* var,
                    const void* beta, const void* gamma, const void* epsilon, std::vector<int>& in_dims, void* out_data)
{
    const int BATCH = in_dims[0];
    const int CH = in_dims[1];
    const int HEIGHT = in_dims[2];
    const int WIDTH = in_dims[3];

    std::cout << "Launcher for FIM_BN" << std::endl;

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_mean = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_var = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_beta = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_gamma = FimCreateBo(1, 1, CH, 1, FIM_FP16, MEM_TYPE_HOST);
    double host_epsilon = 0.0;

    FimBo* preloaded_fim_input = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(WIDTH, HEIGHT, CH, BATCH, FIM_FP16, MEM_TYPE_FIM);

    FimCopyMemory((void*)host_input->data, (void*)inp_data, sizeof(half) * N, DEVICE_TO_HOST);

    /* __FIM_API__ call : Preload input data on FIM memory */
    FimConvertDataLayout(preloaded_fim_input, host_input, OP_BN);

    FimCopyMemory(host_mean->data, (void*)mean, sizeof(half) * CH, DEVICE_TO_HOST);
    FimCopyMemory(host_var->data, (void*)var, sizeof(half) * CH, DEVICE_TO_HOST);
    FimCopyMemory(host_gamma->data, (void*)gamma, sizeof(half) * CH, DEVICE_TO_HOST);
    FimCopyMemory(host_beta->data, (void*)beta, sizeof(half) * CH, DEVICE_TO_HOST);
    FimCopyMemory((void*)&host_epsilon, (void*)epsilon, sizeof(double), DEVICE_TO_HOST);

    std::cout << "Calling FIMExecuteBN" << std::endl;
    /* __FIM_API__ call : Execute FIM kernel */
    FimExecuteBN(device_output, preloaded_fim_input, host_beta, host_gamma, host_mean, host_var, host_epsilon);
    FimSynchronize();

    FimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(half) * N, FIM_TO_HOST);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_mean);
    FimDestroyBo(host_var);
    FimDestroyBo(host_beta);
    FimDestroyBo(host_gamma);
    FimDestroyBo(preloaded_fim_input);
    FimDestroyBo(device_output);
}

torch::Tensor py_fim_bn(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor offset,
                        torch::Tensor scale, torch::Tensor epsilon)
{
    const int N = input.numel();   // compute num elements in tensor
    const int DIMS = input.dim();  // compute num_dims of a tensor

    auto inp_data = torch::flatten(input).data_ptr<at::Half>();
    auto mean_data = torch::flatten(mean).data_ptr<at::Half>();
    auto var_data = torch::flatten(var).data_ptr<at::Half>();

    auto offset_data = torch::flatten(offset).data_ptr<at::Half>();
    auto scale_data = torch::flatten(scale).data_ptr<at::Half>();
    auto eps = torch::flatten(epsilon).data_ptr<double>();

    torch::Tensor output = torch::zeros_like(input);
    auto out_data = output.data_ptr<at::Half>();

    std::vector<int> inp_dims;
    for (int i = 0; i < DIMS; i++) {
        inp_dims.push_back(input.sizes()[i]);
    }

    KernelLauncher(inp_data, N, DIMS, mean_data, var_data, offset_data, scale_data, eps, inp_dims, out_data);

    return output;
}

static auto registry = torch::RegisterOperators("custom_ops::py_fim_bn", &py_fim_bn);
