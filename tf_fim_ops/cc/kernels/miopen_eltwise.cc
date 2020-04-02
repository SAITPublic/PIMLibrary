#include <miopen/miopen.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "fim_runtime_api.h"
#include <iostream>

using namespace tensorflow;  // NOLINT(build/namespaces)

static const int align = 64 * 1024;

void KernelLauncher(const void* a_data, const void* b_data , const int N, void* c_data, int32 op)
{
    const int LENGTH = ((N + align - 1) / align ) * align;

    std::cout << "Launcher for MIopen" << std::endl;
    miopenTensorDescriptor_t a_desc, b_desc, c_desc;

    miopenCreateTensorDescriptor(&a_desc);
    miopenCreateTensorDescriptor(&b_desc);
    miopenCreateTensorDescriptor(&c_desc);

    std::vector<int> a_len = {LENGTH};
    std::vector<int> b_len = {LENGTH};
    std::vector<int> c_len = {LENGTH};

    miopenSetTensorDescriptor(a_desc, miopenHalf, 1, a_len.data(), nullptr);
    miopenSetTensorDescriptor(b_desc, miopenHalf, 1, b_len.data(), nullptr);
    miopenSetTensorDescriptor(c_desc, miopenHalf, 1, c_len.data(), nullptr);
    float alpha_0 = 1;
    float alpha_1 = 1;
    float beta = 0;
    miopenHandle_t handle;
    miopenCreate(&handle);
    if(op == 0)
        miopenOpTensor(handle, miopenTensorOpAdd, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                       c_data);
    else
        miopenOpTensor(handle, miopenTensorOpMul, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                       c_data);
    miopenDestroy(handle);
}

class MiopenEltwiseOp : public OpKernel {
 public:
  explicit MiopenEltwiseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<Eigen::half>();

    const Tensor& input_tensor1 = context->input(1);
    auto input1 = input_tensor1.flat<Eigen::half>();

    const Tensor& input_tensor2 = context->input(2);
    auto op = input_tensor2.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<Eigen::half>();

    const int N = input.size();

    // Call kernel
    KernelLauncher(input.data(),input1.data(),N,output.data(),op.data()[0]);
  }
};

REGISTER_KERNEL_BUILDER(Name("MiopenEltwise").Device(DEVICE_GPU), MiopenEltwiseOp);
