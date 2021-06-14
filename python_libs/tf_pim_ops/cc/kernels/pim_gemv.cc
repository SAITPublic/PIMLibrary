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

void KernelLauncher(const void* i_data, const void* w_data, const int num_batch, const int IN_LENGTH,
                    const int OUT_LENGTH, void* o_data, int reorder)
{
    DLOG(INFO) << "Launcher for PIM_Gemv";

    //    /* __PIM_API__ call : Initialize PimRuntime */
    //    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    PimDesc* pim_desc = PimCreateDesc(num_batch, 1, OUT_LENGTH, IN_LENGTH, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    // Copy , incement using descriptors bshape.w
    for (int i = 0; i < num_batch; i++) {
        PimCopyMemory((void*)(static_cast<half*>(host_input->data) + i * pim_desc->bshape.w),
                      (void*)(static_cast<const half*>(i_data) + i * IN_LENGTH), sizeof(half) * IN_LENGTH,
                      DEVICE_TO_HOST);
    }

    // Old ver. PimCopyMemory from tensor weight to Pimbo structure
    // PimCopyMemory((void*)host_weight->data, (void*)w_data, sizeof(half) * IN_LENGTH * OUT_LENGTH, HOST_TO_HOST);

    // Transpose the weight matrix for PIM spec.
    for (int i = 0; i < IN_LENGTH; i++) {
        for (int j = 0; j < OUT_LENGTH; j++) {
            PimCopyMemory((void*)(static_cast<half*>(host_weight->data) + (j * IN_LENGTH + i)),
                          (void*)(static_cast<const half*>(w_data) + (i * OUT_LENGTH + j)), sizeof(half),
                          DEVICE_TO_HOST);
        }
    }

    /* Initialize the input, weight, output data */
    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    DLOG(INFO) << "Calling PIMExecuteGEMV";
    /* __PIM_API__ call : Execute PIM kernel (GEMV) */
    PimExecuteGemv(device_output, device_input, host_weight);

    PimCopyMemory(o_data, device_output->data, sizeof(half) * num_batch * OUT_LENGTH, DEVICE_TO_HOST);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_output);

    //    /* __PIM_API__ call : Deinitialize PimRuntime */
    //    PimDeinitialize();
}

class PimGemvOp : public OpKernel
{
   public:
    explicit PimGemvOp(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<Eigen::half>();

        const Tensor& input_tensor1 = context->input(1);
        auto input1 = input_tensor1.flat<Eigen::half>();

        int num_batch = input_tensor.dim_size(0);
        int num_rows = input_tensor1.dim_size(0);
        int num_cols = input_tensor1.dim_size(1);

        const Tensor& input_tensor2 = context->input(2);
        int reorder;
        PimCopyMemory((void*)&reorder, (void*)input_tensor2.flat<int32>().data(), sizeof(int), DEVICE_TO_HOST);

        std::cout << "Input Num batches : " << num_batch << std::endl;
        std::cout << "Weight Num inputs : " << num_rows << std::endl;
        std::cout << "Weight Num outputs : " << num_cols << std::endl;

        // Create an output tensor
        Tensor* output_tensor = NULL;
        TensorShape tshape = TensorShape({num_batch, num_cols});

        OP_REQUIRES_OK(context, context->allocate_output(0, tshape, /*input_tensor.shape()*/
                                                         &output_tensor));
        auto output = output_tensor->flat<Eigen::half>();

        // Call kernel
        // num_rows(input) and num_cols(output) should be input like this
        KernelLauncher(input.data(), input1.data(), num_batch, num_rows, num_cols, output.data(), reorder);
    }
};

REGISTER_KERNEL_BUILDER(Name("PimGemv").Device(DEVICE_GPU), PimGemvOp);
