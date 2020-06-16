#include <iostream>
#include "fim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher()
{
    std::cout << "Launcher for FIM_Init" << std::endl;

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();
}

class FimDeinitOp : public OpKernel
{
   public:
    explicit FimDeinitOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override { KernelLauncher(); }
};

REGISTER_KERNEL_BUILDER(Name("FimDeinit").Device(DEVICE_GPU), FimDeinitOp);
