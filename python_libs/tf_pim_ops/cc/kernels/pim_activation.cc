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
#include "hip/hip_fp16.h"
#include "pim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "utility/pim_log.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* inp_data, const int N, void* out_data)
{
    DLOG(INFO) << "Launcher for PIM_Activation";

    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, N, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* pim_input = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    /* __PIM_API__ call : Copy input data on PIM memory */
    PimCopyMemory(pim_input->data, (void*)inp_data, sizeof(half) * N, HOST_TO_PIM);

    DLOG(INFO) << "Calling PIMExecuteRelu";
    /* __PIM_API__ call : Execute PIM kernel (Relu) */
    PimExecuteRelu(device_output, pim_input);

    PimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(half) * N, PIM_TO_HOST);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input);
    PimDestroyDesc(pim_desc);
}

class PimActivationOp : public OpKernel
{
   public:
    explicit PimActivationOp(OpKernelConstruction* context) : OpKernel(context) {}

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

REGISTER_KERNEL_BUILDER(Name("PimActivation").Device(DEVICE_GPU), PimActivationOp);
