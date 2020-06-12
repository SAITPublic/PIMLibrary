#include <iostream>
#include "fim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher()
{
    std::cout << "Launcher for FIM_Init" << std::endl;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);
}

class FimInitOp : public OpKernel
{
   public:
    explicit FimInitOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        KernelLauncher();
    }
};

REGISTER_KERNEL_BUILDER(Name("FimInit").Device(DEVICE_GPU), FimInitOp);
