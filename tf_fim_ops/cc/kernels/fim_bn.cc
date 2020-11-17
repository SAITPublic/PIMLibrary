#include <miopen/miopen.h>
#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "utility/fim_log.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* inp_data, const int N, const int DIMS, const void* mean, const void* var,
                    const void* beta, const void* gamma, const void* epsilon, std::vector<int>& in_dims, void* out_data)
{
    const int BATCH = in_dims[0];
    const int CH = in_dims[1];
    const int HEIGHT = in_dims[2];
    const int WIDTH = in_dims[3];

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

class FimBnOp : public OpKernel
{
   public:
    explicit FimBnOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<Eigen::half>();

        const Tensor& mean_tensor = context->input(1);
        auto mean = mean_tensor.flat<Eigen::half>();

        const Tensor& var_tensor = context->input(2);
        auto var = var_tensor.flat<Eigen::half>();

        const Tensor& beta_tensor = context->input(3);
        auto beta = beta_tensor.flat<Eigen::half>();

        const Tensor& gamma_tensor = context->input(4);
        auto gamma = gamma_tensor.flat<Eigen::half>();

        const Tensor& eps = context->input(5);
        auto epsilon = eps.flat<double>();

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output = output_tensor->template flat<Eigen::half>();

        const int N = input.size();
        const int DIMS = input_tensor.dims();
        std::vector<int> in_dims;

        for (int i = 0; i < DIMS; i++) {
            DLOG(INFO) << "Dim " << input_tensor.dim_size(i);
            in_dims.push_back(input_tensor.dim_size(i));
        }
        // Call kernel
        KernelLauncher(input.data(), N, DIMS, mean.data(), var.data(), beta.data(), gamma.data(), epsilon.data(),
                       in_dims, output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("FimBn").Device(DEVICE_GPU), FimBnOp);
