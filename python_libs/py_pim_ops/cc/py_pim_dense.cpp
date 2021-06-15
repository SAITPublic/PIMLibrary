#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include "hip/hip_fp16.h"
#include "pim_runtime_api.h"

void KernelLauncher(const void* i_data, const void* w_data, const int num_batch, const int IN_LENGTH,
                    const int OUT_LENGTH, void* o_data)
{
    std::cout << "Launcher for PIM_Dense " << std::endl;

    PimBo* dev_in = nullptr;
    PimBo* pre_wei = nullptr;
    PimBo* dev_out = nullptr;
    PimBo t_dev_in;
    PimBo t_dev_out;

    uint64_t weight_key = reinterpret_cast<uint64_t>(w_data);

    PimGemvBundle* bundle = PimFindGemvBundle(weight_key);
    if (bundle == nullptr) {
        PimDesc* pim_desc = PimCreateDesc(num_batch, 1, OUT_LENGTH, IN_LENGTH, PIM_FP16, OP_GEMV);
        PimBo* weight = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);
        pre_wei = PimCreateBo(pim_desc, MEM_TYPE_PIM, GEMV_WEIGHT);
        dev_in = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
        dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

        // Transpose the weight matrix for PIM spec.
        for (int i = 0; i < IN_LENGTH; i++) {
            for (int j = 0; j < OUT_LENGTH; j++) {
                PimCopyMemory((void*)(static_cast<half*>(weight->data) + (j * IN_LENGTH + i)),
                              (void*)(static_cast<const half*>(w_data) + (i * OUT_LENGTH + j)), sizeof(half),
                              DEVICE_TO_PIM);
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
                          DEVICE_TO_PIM);
        }
    } else {
        dev_in->data = (void*)i_data;
    }

    t_dev_out = *dev_out;
    t_dev_out.data = o_data;
    PimExecuteGemv(&t_dev_out, dev_in, pre_wei);
    PimSynchronize();
    // Not setting this causes a crash in pim_deinit()
    dev_in->data = dev_data_ptr;
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

    std::cout << "Input Dims :" << num_dims << std::endl;
    std::cout << "Input Num batches : " << num_batch << std::endl;
    std::cout << "Weight Num inputs : " << num_rows << std::endl;
    std::cout << "Weight Num outputs : " << num_cols << std::endl;

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
