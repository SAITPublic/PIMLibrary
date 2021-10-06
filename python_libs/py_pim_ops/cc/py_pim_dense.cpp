#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "pim_runtime_api.h"
#include "utility/pim_log.h"

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
    PimDesc* pim_desc = PimCreateDesc(num_batch, 1, OUT_LENGTH, IN_LENGTH, PIM_FP16, OP_GEMV);
    PimBo* dev_weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT, const_cast<void*>(w_data));
    dev_in = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

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

    PimExecuteGemv(&t_dev_out, dev_in, dev_weight);

    // Not setting this causes a crash in pim_deinit()
    dev_in->data = dev_data_ptr;

    PimDestroyBo(dev_weight);
    PimDestroyDesc(pim_desc);
}

torch::Tensor py_pim_dense(torch::Tensor input, torch::Tensor weights, torch::Tensor bias_flag, torch::Tensor bias)
{
    auto input_data = torch::flatten(input).data_ptr<at::Half>();
    // Weight matrix needs to be transposed as FIM kernel expects shape to be similar to that of
    // a tensorflow weight matrix.
    auto weights_data = torch::flatten(weights).data_ptr<at::Half>();
    auto bias_data = torch::flatten(bias).data_ptr<at::Half>();

    int num_iters = 1;
    int num_dims = input.dim();
    int num_batch = input.sizes()[0];
    int num_rows = weights.sizes()[0];
    int num_cols = weights.sizes()[1];
    auto dev = input.device();
    torch::Tensor output_tensor;
    if (num_dims > 3) {
        std::cout << "Currently only upto 3 dim inputs are supported " << std::endl;
        return output_tensor;
    }

    int has_bias = 0;
    has_bias = bias_flag.item<int>();
    ;

    DLOG(INFO) << "Input Dims :" << num_dims;
    DLOG(INFO) << "Input Num batches : " << num_batch;
    DLOG(INFO) << "Weight Num inputs : " << num_rows;
    DLOG(INFO) << "Weight Num outputs : " << num_cols;

    if (num_dims == 3) {
        num_iters = input.sizes()[1];
        output_tensor = torch::zeros({num_batch, num_iters, num_cols},
                                     torch::dtype(torch::kF16).device(input.device().type(), input.device().index()));
    } else {
        output_tensor = torch::zeros({num_batch, num_cols},
                                     torch::dtype(torch::kF16).device(input.device().type(), input.device().index()));
    }
    auto output = torch::flatten(output_tensor).data_ptr<at::Half>();
    int offset_row = 0;
    int offset_col = 0;

    for (int i = 0; i < num_iters; i++) {
        KernelLauncher(input_data + offset_row, weights_data, num_batch, num_rows, num_cols, output + offset_col);
        offset_row += num_rows * num_batch;
        offset_col += num_cols * num_batch;
    }
    if (has_bias) {
        output_tensor = torch::add(output_tensor, bias);
    }
    return output_tensor;
}
static auto registry = torch::RegisterOperators("custom_ops::py_pim_dense", &py_pim_dense);
