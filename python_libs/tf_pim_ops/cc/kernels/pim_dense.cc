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
#include "pim_runtime_api.h"
#include "hip/hip_fp16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "utility/pim_log.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* i_data, const void* w_data, const int num_batch, const int IN_LENGTH,
                    const int OUT_LENGTH, void* o_data)
{
    DLOG(INFO) << "Launcher for PIM_Dense ";

    PimBo* dev_in = nullptr;
    PimBo* pre_wei = nullptr;
    PimBo* dev_out = nullptr;
    PimBo t_dev_in;
    PimBo t_dev_out;

    uint64_t weight_key = reinterpret_cast<uint64_t>(w_data);

    PimGemvBundle* bundle = PimFindGemvBundle(weight_key);
    if (bundle == nullptr) {
        PimDesc* pim_desc = PimCreateDesc(num_batch, 1, OUT_LENGTH, IN_LENGTH, PIM_FP16, OP_GEMV);
        PimBo* weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
        pre_wei = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);
        dev_in = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
        dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

        // Transpose the weight matrix for PIM spec.
        for (int i = 0; i < IN_LENGTH; i++) {
            for (int j = 0; j < OUT_LENGTH; j++) {
                PimCopyMemory((void*)(static_cast<half*>(weight->data) + (j * IN_LENGTH + i)),
                              (void*)(static_cast<const half*>(w_data) + (i * OUT_LENGTH + j)), sizeof(half),
                              DEVICE_TO_DEVICE);
            }
        }

        PimConvertDataLayout(pre_wei, weight, OP_GEMV);

        bundle = PimCreateGemvBundle(dev_in, pre_wei, dev_out);
        PimInsertGemvBundle(weight_key, bundle);

        PimDestroyBo(weight);
        PimDestroyDesc(pim_desc);
    } else {
        dev_in = bundle->in;
        pre_wei = bundle->wei;
        dev_out = bundle->out;
    }

    void* dev_data_ptr = dev_in->data;
    if (num_batch > 1) {
        for (int i = 0; i < num_batch; i++) {
            PimCopyMemory((void*)(static_cast<half*>(dev_in->data) + i * dev_in->bshape.w),
                          (void*)(static_cast<const half*>(i_data) + i * IN_LENGTH), sizeof(half) * IN_LENGTH,
                          DEVICE_TO_DEVICE);
        }
    } else {
        dev_in->data = (void*)i_data;
    }

    t_dev_out = *dev_out;
    t_dev_out.data = o_data;

    PimExecuteGemv(&t_dev_out, dev_in, pre_wei);

    // Not setting this causes a crash in pim_deinit()
    dev_in->data = dev_data_ptr;
}

class PimDenseOp : public OpKernel
{
   public:
    explicit PimDenseOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<Eigen::half>();

        const Tensor& input_tensor1 = context->input(1);
        auto input1 = input_tensor1.flat<Eigen::half>();

        const Tensor& input_tensor2 = context->input(2);
        auto bias = input_tensor2.flat<Eigen::half>();

        int num_iters = 1;
        int num_dims = input_tensor.dims();
        int num_batch = input_tensor.dim_size(0);
        int num_rows = input_tensor1.dim_size(0);
        int num_cols = input_tensor1.dim_size(1);

        if (num_dims > 3) {
            DLOG(ERROR) << "Currently only upto 3 dim inputs are supported " << std::endl;
            return;
        }

        int has_bias = 0;
        const Tensor& input_tensor3 = context->input(3);
//        has_bias = input_tensor3.flat<int32>().data()[0];

        DLOG(INFO) << "Input Dims :" << num_dims;
        DLOG(INFO) << "Input Num batches : " << num_batch;
        DLOG(INFO) << "Weight Num inputs : " << num_rows;
        DLOG(INFO) << "Weight Num outputs : " << num_cols;

        // Create an output tensor
        Tensor* output_tensor = NULL;
        TensorShape tshape = TensorShape({num_batch, num_cols});

        if (num_dims == 3) {
            num_iters = input_tensor.dim_size(1);
            tshape = TensorShape({num_batch, num_iters, num_cols});
        }

        OP_REQUIRES_OK(context, context->allocate_output(0, tshape, &output_tensor));
        auto output = output_tensor->flat<Eigen::half>();

        int offset_row = 0;
        int offset_cols = 0;
        for (int i = 0; i < num_iters; i++) {
            KernelLauncher(input.data() + offset_row, input1.data(), num_batch, num_rows, num_cols,
                           output.data() + offset_cols);
            offset_row += num_rows * num_batch;
            offset_cols += num_cols * num_batch;
        }

        if (has_bias) {
            for (int i = 0; i < num_cols * num_batch * num_iters; i++) output.data()[i] += bias.data()[i % num_cols];

            // Todo: Can we use pim.eltwise
            // PimGemvBundle* bundle = PimFindGemvBundle(weight_key);
            // PimBo* device_output = bundle->out;
            // KernelLauncherAdd(bias.data(), output.data(), num_cols * num_batch , output.data());
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("PimDense").Device(DEVICE_GPU), PimDenseOp);
