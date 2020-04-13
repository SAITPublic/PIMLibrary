#include <miopen/miopen.h>
#include <iostream>
#include "fim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

static const int align = 64 * 1024;

void KernelLauncher(const void* i_data, const int N, void* o_data)
{
    const int LENGTH = ((N + align - 1) / align) * align;

    std::cout << "Launcher for MIopen Activation" << std::endl;
    miopenTensorDescriptor_t i_desc, o_desc;
    miopenCreateTensorDescriptor(&i_desc);
    miopenCreateTensorDescriptor(&o_desc);

    std::vector<int> i_len = {LENGTH};
    std::vector<int> o_len = {LENGTH};

    miopenSetTensorDescriptor(i_desc, miopenHalf, 1, i_len.data(), nullptr);
    miopenSetTensorDescriptor(o_desc, miopenHalf, 1, o_len.data(), nullptr);

    static float alpha = 1.0;
    static float beta = 0.0;
    miopenActivationDescriptor_t a_desc;
    miopenCreateActivationDescriptor(&a_desc);
    miopenSetActivationDescriptor(a_desc, miopenActivationRELU, alpha, beta, 1.0);

    miopenHandle_t handle;
    miopenCreate(&handle);
    // Todo , what does alpha , beta mean , check datatype also
    miopenActivationForward(handle, a_desc, (void*)&alpha, i_desc, i_data, (void*)&beta, o_desc, o_data);

    miopenDestroyActivationDescriptor(a_desc);
    miopenDestroy(handle);
}

class MiopenActivationOp : public OpKernel
{
   public:
    explicit MiopenActivationOp(OpKernelConstruction* context) : OpKernel(context) {}

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

REGISTER_KERNEL_BUILDER(Name("MiopenActivation").Device(DEVICE_GPU), MiopenActivationOp);
