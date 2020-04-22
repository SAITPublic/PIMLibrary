#include <miopen/miopen.h>
#include <iostream>
#include "fim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

static const int align = 64 * 1024;

void KernelLauncher(const void* i_data, const int N, const int DIMS, const void* mean, const void* var,
                    const void* offset, const void* scale, std::vector<int>& len, void* o_data)
{
    const int LENGTH = ((N + align - 1) / align) * align;

    std::cout << "Launcher for MIopen Activation" << std::endl;
    miopenTensorDescriptor_t i_desc, o_desc, w_desc;
    miopenCreateTensorDescriptor(&i_desc);
    miopenCreateTensorDescriptor(&o_desc);
    miopenCreateTensorDescriptor(&w_desc);

    miopenSetTensorDescriptor(i_desc, miopenHalf, DIMS, len.data(), nullptr);
    miopenSetTensorDescriptor(o_desc, miopenHalf, DIMS, len.data(), nullptr);

    // Todo: Add a case for 2D also
    std::vector<int> w_len = {1, len[1], 1, 1};
    miopenSetTensorDescriptor(w_desc, miopenHalf, 4, w_len.data(), nullptr);

    miopenHandle_t handle;
    miopenCreate(&handle);

    miopenBatchNormMode_t mode;
    if (DIMS == 2)
        mode = miopenBNPerActivation;
    else
        mode = miopenBNSpatial;

    float alpha = 1.0;
    float beta = 0.0;
    double epsilon = 1e-3;

    void* out;
    hipMalloc(&out, sizeof(half) * N);

    miopenBatchNormalizationForwardInference(handle, mode, &alpha, &beta, i_desc, i_data, o_desc, out, w_desc,
                                             (void*)scale, (void*)offset, (void*)mean, (void*)var, epsilon);

    // Todo check return value
    auto status = hipMemcpy(o_data, out, sizeof(half) * N, hipMemcpyDeviceToHost);

    hipFree(out);
    miopenDestroyTensorDescriptor(i_desc);
    miopenDestroyTensorDescriptor(o_desc);
    miopenDestroyTensorDescriptor(w_desc);
    miopenDestroy(handle);
}

class MiopenBnOp : public OpKernel
{
   public:
    explicit MiopenBnOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<Eigen::half>();

        const Tensor& mean_tensor = context->input(1);
        auto mean = mean_tensor.flat<Eigen::half>();

        const Tensor& var_tensor = context->input(2);
        auto variance = var_tensor.flat<Eigen::half>();

        const Tensor& offset_tensor = context->input(3);
        auto offset = offset_tensor.flat<Eigen::half>();

        const Tensor& scale_tensor = context->input(4);
        auto scale = scale_tensor.flat<Eigen::half>();

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output = output_tensor->template flat<Eigen::half>();

        const int N = input.size();
        const int DIMS = input_tensor.dims();
        std::vector<int> len;

        for (int i = 0; i < DIMS; i++) {
            std::cout << "Dim " << input_tensor.dim_size(i);
            len.push_back(input_tensor.dim_size(i));
        }
        std::cout << "input" << input.data()[0] << std::endl;
        std::cout << "mean" << mean.data()[0] << std::endl;
        std::cout << "var" << variance.data()[0] << std::endl;
        std::cout << "offset" << offset.data()[0] << std::endl;
        std::cout << "scale" << scale.data()[0];

        // Call kernel
        KernelLauncher(input.data(), N, DIMS, mean.data(), variance.data(), offset.data(), scale.data(), len,
                       output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("MiopenBn").Device(DEVICE_GPU), MiopenBnOp);
