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

void KernelLauncher(const void* inp0_data, const void* inp1_data, int N, int is_scalar, void* out_data, int32 op)
{
    DLOG(INFO) << "Launcher for PIM_Eltwise";

    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, N, PIM_FP16);

    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* pim_input1 = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    PimCopyMemory((void*)pim_input1->data, (void*)inp1_data, sizeof(uint16_t) * N, HOST_TO_PIM);

    if (is_scalar == 1) {
        uint16_t pim_input0;

        PimCopyMemory((void*)&pim_input0, (void*)inp0_data, sizeof(uint16_t), DEVICE_TO_HOST);

        if (op == 0) {
            DLOG(INFO) << "Calling PIMExecuteAdd";
            PimExecuteAdd(device_output, (void*)&pim_input0, pim_input1);
        } else {
            DLOG(INFO) << "Calling PIMExecuteMul";
            PimExecuteMul(device_output, (void*)&pim_input0, pim_input1);
        }
    } else {
        PimBo* pim_input0 = PimCreateBo(pim_desc, MEM_TYPE_PIM);

        PimCopyMemory((void*)pim_input0->data, (void*)inp0_data, sizeof(uint16_t) * N, HOST_TO_PIM);

        if (op == 0) {
            DLOG(INFO) << "Calling PIMExecuteAdd";
            PimExecuteAdd(device_output, pim_input0, pim_input1);
        } else {
            DLOG(INFO) << "Calling PIMExecuteMul";
            PimExecuteMul(device_output, pim_input0, pim_input1);
        }

        PimDestroyBo(pim_input0);
    }

    PimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(uint16_t) * N, PIM_TO_HOST);

    /* __PIM_API__ call : Free memory */
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input1);
    PimDestroyDesc(pim_desc);
}

class PimEltwiseOp : public OpKernel
{
   public:
    explicit PimEltwiseOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor0 = context->input(0);
        auto input0 = input_tensor0.flat<Eigen::half>();

        const Tensor& input_tensor1 = context->input(1);
        auto input1 = input_tensor1.flat<Eigen::half>();

        const Tensor& input_tensor2 = context->input(2);
        int op;
        PimCopyMemory((void*)&op, (void*)input_tensor2.flat<int32>().data(), sizeof(int), DEVICE_TO_HOST);

        int N0 = input0.size();
        int N1 = input1.size();
        int wi = -1;  // flag to check which input tensor is scalar
        int is_scalar = 0;

        // Create an output tensor
        Tensor* output_tensor = nullptr;

        if (N0 != N1) {
            // if num elems are not equal then check if one of them is a scalar.
            if (N0 != 1 && N1 != 1) {
                DLOG(INFO) << "num elems in both tensors are not same and neither of them is a scalar \n";
                // TODO
                // throw error
            } else if (N0 == 1)
                wi = 0;
            else
                wi = 1;
        } else {
            // TODO
            // if num elements are equal check if shapes are equal.
            DLOG(INFO) << "check for shapes \n";
        }
        if (wi != -1) is_scalar = 1;  // this means one of the inp tensors is a scalar

        if (wi == 1) {
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor0.shape(), &output_tensor));
            auto output = output_tensor->template flat<Eigen::half>();
            // Call kernel
            KernelLauncher(input1.data(), input0.data(), N0, is_scalar, output.data(), op);

        } else {
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(), &output_tensor));
            auto output = output_tensor->template flat<Eigen::half>();
            // Call kernel
            KernelLauncher(input0.data(), input1.data(), N1, is_scalar, output.data(), op);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("PimEltwise").Device(DEVICE_GPU), PimEltwiseOp);
