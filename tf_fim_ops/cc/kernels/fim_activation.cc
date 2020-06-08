#include <iostream>
#include "fim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* inp_data, const int N, void* out_data)
{
    std::cout << "Launcher for FIM_Activation" << std::endl;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimDesc* fim_desc = FimCreateDesc(1, 1, 1, N, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* fim_input = FimCreateBo(fim_desc, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_FIM);

    /* __FIM_API__ call : Copy input data on FIM memory */
    FimCopyMemory((void*)host_input->data, (void*)inp_data, sizeof(half) * N, HOST_TO_HOST);
    FimCopyMemory(fim_input, host_input, HOST_TO_FIM);

    std::cout << "Calling FIMExecuteRelu" << std::endl;
    /* __FIM_API__ call : Execute FIM kernel (Relu) */
    FimExecuteRelu(device_output, fim_input);

    FimCopyMemory(host_output, device_output, FIM_TO_HOST);
    FimCopyMemory((void*)out_data, (void*)host_output->data, sizeof(half) * N, HOST_TO_HOST);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input);
    FimDestroyBo(host_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);
    FimDestroyDesc(fim_desc);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();
}

class FimActivationOp : public OpKernel
{
   public:
    explicit FimActivationOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<Eigen::half>();

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output = output_tensor->template flat<Eigen::half>();

        const int N = input.size();

        // Call kernel
        KernelLauncher(input.data(), N, output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("FimActivation").Device(DEVICE_GPU), FimActivationOp);
