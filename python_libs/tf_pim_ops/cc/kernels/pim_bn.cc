/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <miopen/miopen.h>
#include <iostream>
#include "hip/hip_fp16.h"
#include "pim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "utility/pim_log.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* inp_data, const int N, const int DIMS, const void* mean, const void* var,
                    const void* beta, const void* gamma, const void* epsilon, std::vector<int>& in_dims, void* out_data)
{
    const int BATCH = in_dims[0];
    const int CH = in_dims[1];
    const int HEIGHT = in_dims[2];
    const int WIDTH = in_dims[3];

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_mean = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_var = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_beta = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    PimBo* host_gamma = PimCreateBo(1, 1, CH, 1, PIM_FP16, MEM_TYPE_HOST);
    double host_epsilon = 0.0;

    PimBo* preloaded_pim_input = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(WIDTH, HEIGHT, CH, BATCH, PIM_FP16, MEM_TYPE_PIM);

    PimCopyMemory((void*)host_input->data, (void*)inp_data, sizeof(half) * N, DEVICE_TO_HOST);

    /* __PIM_API__ call : Preload input data on PIM memory */
    // PimConvertDataLayout(preloaded_pim_input, host_input, OP_BN);

    PimCopyMemory(host_mean->data, (void*)mean, sizeof(half) * CH, DEVICE_TO_HOST);
    PimCopyMemory(host_var->data, (void*)var, sizeof(half) * CH, DEVICE_TO_HOST);
    PimCopyMemory(host_gamma->data, (void*)gamma, sizeof(half) * CH, DEVICE_TO_HOST);
    PimCopyMemory(host_beta->data, (void*)beta, sizeof(half) * CH, DEVICE_TO_HOST);
    PimCopyMemory((void*)&host_epsilon, (void*)epsilon, sizeof(double), DEVICE_TO_HOST);

    /* __PIM_API__ call : Execute PIM kernel */
    PimExecuteBN(device_output, host_input, host_beta, host_gamma, host_mean, host_var, host_epsilon);
    PimSynchronize();

    PimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(half) * N, PIM_TO_HOST);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(host_input);
    PimDestroyBo(host_mean);
    PimDestroyBo(host_var);
    PimDestroyBo(host_beta);
    PimDestroyBo(host_gamma);
    PimDestroyBo(preloaded_pim_input);
    PimDestroyBo(device_output);
}

class PimBnOp : public OpKernel
{
   public:
    explicit PimBnOp(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<Eigen::half>();

        const Tensor& mean_tensor = context->input(1);
        auto mean = mean_tensor.flat<Eigen::half>();

        const Tensor& var_tensor = context->input(2);
        auto var = var_tensor.flat<Eigen::half>();

        const Tensor& beta_tensor = context->input(3);
        auto beta = beta_tensor.flat<Eigen::half>();

        const Tensor& gamma_tensor = context->input(4);
        auto gamma = gamma_tensor.flat<Eigen::half>();

        const Tensor& eps = context->input(5);
        auto epsilon = eps.flat<double>();

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output = output_tensor->template flat<Eigen::half>();

        const int N = input.size();
        const int DIMS = input_tensor.dims();
        std::vector<int> in_dims;

        for (int i = 0; i < DIMS; i++) {
            DLOG(INFO) << "Dim " << input_tensor.dim_size(i);
            in_dims.push_back(input_tensor.dim_size(i));
        }
        // Call kernel
        KernelLauncher(input.data(), N, DIMS, mean.data(), var.data(), beta.data(), gamma.data(), epsilon.data(),
                       in_dims, output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("PimBn").Device(DEVICE_GPU), PimBnOp);
