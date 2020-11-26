#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "utility/fim_log.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher(const void* i_data, const void* w_data, const int num_batch, const int IN_LENGTH,
                    const int OUT_LENGTH, void* o_data)
{
    DLOG(INFO) << "Launcher for FIM_Dense ";

    FimBo* dev_in = nullptr;
    FimBo* pre_wei = nullptr;
    FimBo* dev_out = nullptr;
    FimBo t_dev_in;
    FimBo t_dev_out;

    uint64_t weight_key = reinterpret_cast<uint64_t>(w_data);

    FimGemvBundle* bundle = FimFindGemvBundle(weight_key);
    if (bundle == nullptr) {
        FimDesc* fim_desc = FimCreateDesc(num_batch, 1, OUT_LENGTH, IN_LENGTH, FIM_FP16, OP_GEMV);
        FimBo* weight = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
        pre_wei = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
        dev_in = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
        dev_out = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

        // Transpose the weight matrix for FIM spec.
        for (int i = 0; i < IN_LENGTH; i++) {
            for (int j = 0; j < OUT_LENGTH; j++) {
                FimCopyMemory((void*)(static_cast<half*>(weight->data) + (j * IN_LENGTH + i)),
                              (void*)(static_cast<const half*>(w_data) + (i * OUT_LENGTH + j)), sizeof(half),
                              DEVICE_TO_DEVICE);
            }
        }

        FimConvertDataLayout(pre_wei, weight, OP_GEMV);

        bundle = FimCreateGemvBundle(dev_in, pre_wei, dev_out);
        FimInsertGemvBundle(weight_key, bundle);

        FimDestroyBo(weight);
        FimDestroyDesc(fim_desc);
    } else {
        dev_in = bundle->in;
        pre_wei = bundle->wei;
        dev_out = bundle->out;
    }

    void* dev_data_ptr = dev_in->data;
    if (num_batch > 1) {
        for (int i = 0; i < num_batch; i++) {
            FimCopyMemory((void*)(static_cast<half*>(dev_in->data) + i * dev_in->bshape.w),
                          (void*)(static_cast<const half*>(i_data) + i * IN_LENGTH), sizeof(half) * IN_LENGTH,
                          DEVICE_TO_DEVICE);
        }
    } else {
        dev_in->data = (void*)i_data;
    }

    t_dev_out = *dev_out;
    t_dev_out.data = o_data;

    FimExecuteGemv(&t_dev_out, dev_in, pre_wei);

    // Not setting this causes a crash in fim_deinit()
    dev_in->data = dev_data_ptr;
}

class FimDenseOp : public OpKernel
{
   public:
    explicit FimDenseOp(OpKernelConstruction* context) : OpKernel(context) {}

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
        has_bias = input_tensor3.flat<int32>().data()[0];

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

            // Todo: Can we use fim.eltwise
            // FimGemvBundle* bundle = FimFindGemvBundle(weight_key);
            // FimBo* device_output = bundle->out;
            // KernelLauncherAdd(bias.data(), output.data(), num_cols * num_batch , output.data());
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("FimDense").Device(DEVICE_GPU), FimDenseOp);
