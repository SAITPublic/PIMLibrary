#include <iostream>
#include "fim_runtime_api.h"
#include "half.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* i_data, const void* w_data, const int num_batch, const int IN_LENGTH,
                    const int OUT_LENGTH, void* o_data, int reorder)
{
    std::cout << "Launcher for FIM_Gemv" << std::endl;

    //    /* __FIM_API__ call : Initialize FimRuntime */
    //    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimDesc* fim_desc = FimCreateDesc(num_batch, 1, OUT_LENGTH, IN_LENGTH, FIM_FP16);
    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    FimBo* host_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* host_reordered_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* device_input = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    FimBo* preloaded_weight = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    // Copy , incement using descriptors bshape.w
    for (int i = 0; i < num_batch; i++) {
        memcpy(static_cast<half*>(host_input->data) + i * fim_desc->bshape.w,
               static_cast<const half*>(i_data) + i * IN_LENGTH, sizeof(half) * IN_LENGTH);
    }
    FimCopyMemory((void*)host_weight->data, (void*)w_data, sizeof(half) * IN_LENGTH * OUT_LENGTH, HOST_TO_HOST);

    /* Initialize the input, weight, output data */
    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    if (reorder) {
        std::cout << "Reordering" << std::endl;
        FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
        FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    } else {
        FimCopyMemory(preloaded_weight, host_weight, HOST_TO_DEVICE);
    }

    std::cout << "Calling FIMExecuteGEMV" << std::endl;
    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGEMV(device_output, device_input, preloaded_weight);

    FimCopyMemory(o_data, device_output->data, sizeof(half) * num_batch * OUT_LENGTH, DEVICE_TO_HOST);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);

    //    /* __FIM_API__ call : Deinitialize FimRuntime */
    //    FimDeinitialize();
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

        int num_batch = input_tensor.dim_size(0);
        int num_rows = input_tensor1.dim_size(0);
        int num_cols = input_tensor1.dim_size(1);

        const Tensor& input_tensor2 = context->input(2);
        auto reorder = input_tensor2.flat<int32>();

        std::cout << "Input Num batches : " << num_batch << std::endl;
        std::cout << "Weight Num rows : " << num_rows << std::endl;
        std::cout << "Weight Num cols : " << num_cols << std::endl;

        // Create an output tensor
        Tensor* output_tensor = NULL;
        TensorShape tshape = TensorShape({num_batch, num_cols});

        OP_REQUIRES_OK(context, context->allocate_output(0, tshape, /*input_tensor.shape()*/
                                                         &output_tensor));
        auto output = output_tensor->flat<Eigen::half>();

        // Call kernel
        KernelLauncher(input.data(), input1.data(), num_batch, num_rows, num_cols, output.data(), reorder.data()[0]);
    }
};

REGISTER_KERNEL_BUILDER(Name("FimGemv").Device(DEVICE_GPU), FimGemvOp);
