#include <torch/script.h>
#include <torch/torch.h>
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "pim_runtime_api.h"

void KernelLauncher(const void* inp_data, const int N, const int DIMS, const void* mean, const void* var,
                    const void* beta, const void* gamma, const void* epsilon, std::vector<int>& in_dims, void* out_data)
{
    const int BATCH = in_dims[0];
    const int CH = in_dims[1];
    const int HEIGHT = in_dims[2];
    const int WIDTH = in_dims[3];

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_var = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    double host_epsilon = 0.0;

    /* __PIM_API__ call : Preload input data on PIM memory */
    // PimConvertDataLayout(preloaded_pim_input, host_input, OP_BN);

    PimCopyMemory((void*)pim_input->data, (void*)inp_data, sizeof(half) * N, DEVICE_TO_PIM);
    PimCopyMemory(host_mean->data, (void*)mean, sizeof(half) * CH, DEVICE_TO_HOST);
    PimCopyMemory(host_var->data, (void*)var, sizeof(half) * CH, DEVICE_TO_HOST);
    PimCopyMemory(host_gamma->data, (void*)gamma, sizeof(half) * CH, DEVICE_TO_HOST);
    PimCopyMemory(host_beta->data, (void*)beta, sizeof(half) * CH, DEVICE_TO_HOST);
    PimCopyMemory((void*)&host_epsilon, (void*)epsilon, sizeof(double), DEVICE_TO_HOST);

    /* __PIM_API__ call : Execute PIM kernel */
    PimExecuteBN(device_output, pim_input, host_beta, host_gamma, host_mean, host_var, host_epsilon);
    PimSynchronize();

    PimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(half) * N, HOST_TO_DEVICE);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_mean);
    PimDestroyBo(host_var);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(pim_input);
    PimDestroyBo(device_output);
}

torch::Tensor py_pim_bn(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor offset,
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

static auto registry = torch::RegisterOperators("custom_ops::py_pim_bn", &py_pim_bn);
