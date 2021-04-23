/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "utility/fim_log.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* inp_data, const int N, void* out_data)
{
    DLOG(INFO) << "Launcher for FIM_Activation";

    FimDesc* fim_desc = FimCreateDesc(1, 1, 1, N, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* fim_input = FimCreateBo(fim_desc, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_FIM);

    /* __FIM_API__ call : Copy input data on FIM memory */
    FimCopyMemory(fim_input->data, (void*)inp_data, sizeof(half) * N, HOST_TO_FIM);

    DLOG(INFO) << "Calling FIMExecuteRelu";
    /* __FIM_API__ call : Execute FIM kernel (Relu) */
    FimExecuteRelu(device_output, fim_input);

    FimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(half) * N, FIM_TO_HOST);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input);
    FimDestroyDesc(fim_desc);
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
