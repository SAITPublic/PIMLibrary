#include <iostream>
#include "fim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(string runtimetype, string precision)
{
    std::cout << "Launcher for FIM_Init" << std::endl;
    FimRuntimeType fimruntimetype = RT_TYPE_HIP;
    FimPrecision fimprecision = FIM_FP16;

    if(runtimetype.compare("RT_TYPE_HIP") == 0) {
        fimruntimetype = RT_TYPE_HIP;
    }
    else if(runtimetype.compare("RT_TYPE_OPENCL") == 0) {
        fimruntimetype = RT_TYPE_OPENCL;
    }
    else {
        fimruntimetype = RT_TYPE_HIP;
    }

    if(precision.compare("FIM_FP16") == 0) {
        fimprecision = FIM_FP16;
    }
    else if(precision.compare("FIM_FP16") == 0) {
        fimprecision = FIM_INT8;
    }
    else {
        fimprecision = FIM_FP16;
    }
    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(fimruntimetype, fimprecision);
}

class FimInitOp : public OpKernel
{
    private:
        std::string fimruntimetype;
        std::string fimprecision;

    public:
        explicit FimInitOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("runtimetype", &fimruntimetype));
            OP_REQUIRES_OK(context, context->GetAttr("precision", &fimprecision));
    }

    void Compute(OpKernelContext* context) override {

        KernelLauncher(fimruntimetype, fimprecision);
    }
};

REGISTER_KERNEL_BUILDER(Name("FimInit").Device(DEVICE_GPU), FimInitOp);
