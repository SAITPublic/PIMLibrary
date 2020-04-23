#include <iostream>
#include "fim_runtime_api.h"
#include "half.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* i_data, const void* w_data, const int IN_LENGTH, const int OUT_LENGTH, void* o_data)
{
    std::cout << "Launcher for FIM_Gemv" << std::endl;
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(IN_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_reordered_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_input = FimCreateBo(IN_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* device_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    /* TODO: implement reduce sum for gemv output */
    FimBo* host_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    // FimBo* golden_output = FimCreateBo(OUT_LENGTH * 16, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);

    // Todo: Handle 2 input
    FimCopyMemory((void*)host_input->data, (void*)i_data, 2 * IN_LENGTH, HOST_TO_HOST);
    FimCopyMemory((void*)host_weight->data, (void*)w_data, 2 * IN_LENGTH * OUT_LENGTH, HOST_TO_HOST);

    /* Initialize the input, weight, output data */
    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGEMV(device_output, device_input, preloaded_weight);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    FimCopyMemory((void*)o_data, (void*)host_output->data, 2 * OUT_LENGTH, HOST_TO_HOST);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();
}

class FimGemvOp : public OpKernel
{
   public:
    explicit FimGemvOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<Eigen::half>();

        const Tensor& input_tensor1 = context->input(1);
        auto input1 = input_tensor1.flat<Eigen::half>();

        int num_rows = input_tensor1.dim_size(0);
        int num_cols = input_tensor1.dim_size(1);

        std::cout << "Weight Num rows : " << num_rows << std::endl;
        ;
        std::cout << "Weight Num cols : " << num_cols << std::endl;
        ;

        // Create an output tensor
        Tensor* output_tensor = NULL;
        TensorShape tshape = TensorShape({num_cols});
        OP_REQUIRES_OK(context, context->allocate_output(0, tshape, /*input_tensor.shape()*/
                                                         &output_tensor));
        auto output = output_tensor->flat<Eigen::half>();

        // Call kernel
        KernelLauncher(input.data(), input1.data(), num_rows, num_cols, output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("FimGemv").Device(DEVICE_GPU), FimGemvOp);
