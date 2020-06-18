#include <miopen/miopen.h>
#include <iostream>
#include "fim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* inp0_data, const void* inp1_data, int N, int is_scalar, void* out_data, int32 op)
{
    std::cout << "Launcher for FIM_Eltwise" << std::endl;

    FimDesc* fim_desc = FimCreateDesc(1, 1, 1, N, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input1 = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(fim_desc, MEM_TYPE_HOST);
    FimBo* fim_input1 = FimCreateBo(fim_desc, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_FIM);

    FimCopyMemory((void*)host_input1->data, (void*)inp1_data, sizeof(half) * N, HOST_TO_HOST);
    FimCopyMemory(fim_input1, host_input1, HOST_TO_FIM);

    if (is_scalar == 1) {
        half host_input0;
        half fim_input0;

        FimCopyMemory((void*)&host_input0, (void*)inp0_data, sizeof(half), HOST_TO_HOST);
        FimCopyMemory((void*)&fim_input0, (void*)&host_input0, sizeof(half), HOST_TO_FIM);

        if (op == 0) {
            std::cout << "Calling FIMExecuteAdd" << std::endl;
            FimExecuteAdd(device_output, fim_input0, fim_input1);
        } else {
            std::cout << "Calling FIMExecuteMul" << std::endl;
            FimExecuteMul(device_output, fim_input0, fim_input1);
        }
    } else {
        FimBo* host_input0 = FimCreateBo(fim_desc, MEM_TYPE_HOST);
        FimBo* fim_input0 = FimCreateBo(fim_desc, MEM_TYPE_FIM);

        FimCopyMemory((void*)host_input0->data, (void*)inp0_data, sizeof(half) * N, HOST_TO_HOST);
        FimCopyMemory(fim_input0, host_input0, HOST_TO_FIM);

        if (op == 0) {
            std::cout << "Calling FIMExecuteAdd" << std::endl;
            FimExecuteAdd(device_output, fim_input0, fim_input1);
        } else {
            std::cout << "Calling FIMExecuteMul" << std::endl;
            FimExecuteMul(device_output, fim_input0, fim_input1);
        }

        FimDestroyBo(host_input0);
        FimDestroyBo(fim_input0);
    }

    FimCopyMemory(host_output, device_output, FIM_TO_HOST);
    FimCopyMemory((void*)out_data, (void*)host_output->data, sizeof(half) * N, HOST_TO_HOST);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_input1);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input1);
    FimDestroyDesc(fim_desc);

}

class FimEltwiseOp : public OpKernel
{
   public:
    explicit FimEltwiseOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor0 = context->input(0);
        auto input0 = input_tensor0.flat<Eigen::half>();

        const Tensor& input_tensor1 = context->input(1);
        auto input1 = input_tensor1.flat<Eigen::half>();

        const Tensor& input_tensor2 = context->input(2);
        auto op = input_tensor2.flat<int32>();

        int N0 = input0.size();
        int N1 = input1.size();
        int wi = -1;  // flag to check which input tensor is scalar
        int is_scalar = 0;

        // Create an output tensor
        Tensor* output_tensor = nullptr;

        if (N0 != N1) {
            // if num elems are not equal then check if one of them is a scalar.
            if (N0 != 1 && N1 != 1) {
                std::cout << "num elems in both tensors are not same and neither of them is a scalar \n";
                // TODO
                // throw error
            } else if (N0 == 1)
                wi = 0;
            else
                wi = 1;
        } else {
            // TODO
            // if num elements are equal check if shapes are equal.
            std::cout << "check for shapes \n";
        }
        if (wi != -1) is_scalar = 1;  // this means one of the inp tensors is a scalar

        if (wi == 1) {
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor0.shape(), &output_tensor));
            auto output = output_tensor->template flat<Eigen::half>();
            // Call kernel
            KernelLauncher(input1.data(), input0.data(), N0, is_scalar, output.data(), op.data()[0]);

        } else {
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(), &output_tensor));
            auto output = output_tensor->template flat<Eigen::half>();
            // Call kernel
            KernelLauncher(input0.data(), input1.data(), N1, is_scalar, output.data(), op.data()[0]);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("FimEltwise").Device(DEVICE_GPU), FimEltwiseOp);
