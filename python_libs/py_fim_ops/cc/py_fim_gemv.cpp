#include <iostream>
#include "fim_runtime_api.h"

#include <torch/script.h>
#include <torch/torch.h>
#include "hip/hip_fp16.h"

void KernelLauncher(const void* i_data, const void* w_data, const int num_batch, const int IN_LENGTH,
                    const int OUT_LENGTH, void* o_data, int reorder)
{
    std::cout << "Launcher for FIM_Gemv" << std::endl;

    FimDesc* fim_desc = FimCreateDesc(num_batch, 1, OUT_LENGTH, IN_LENGTH, FIM_FP16, OP_GEMV);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    FimBo* host_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* host_reordered_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* device_input = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    FimBo* preloaded_weight = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    // Copy , incement using descriptors bshape.w
    for (int i = 0; i < num_batch; i++) {
        FimCopyMemory((void*)(static_cast<half*>(host_input->data) + i * fim_desc->bshape.w),
                      (void*)(static_cast<const half*>(i_data) + i * IN_LENGTH), sizeof(half) * IN_LENGTH,
                      DEVICE_TO_HOST);
    }

    // Old ver. FimCopyMemory from tensor weight to Fimbo structure
    // FimCopyMemory((void*)host_weight->data, (void*)w_data, sizeof(half) * IN_LENGTH * OUT_LENGTH, HOST_TO_HOST);

    // Transpose the weight matrix for FIM spec.
    for (int i = 0; i < IN_LENGTH; i++) {
        for (int j = 0; j < OUT_LENGTH; j++) {
            FimCopyMemory((void*)(static_cast<half*>(host_weight->data) + (j * IN_LENGTH + i)),
                          (void*)(static_cast<const half*>(w_data) + (i * OUT_LENGTH + j)), sizeof(half),
                          DEVICE_TO_HOST);
        }
    }

    /* Initialize the input, weight, output data */
    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    if (reorder) {
        std::cout << "Reordering" << std::endl;
        FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
        FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);
    } else {
        FimCopyMemory(preloaded_weight, host_weight, HOST_TO_DEVICE);
    }

    std::cout << "Calling FIMExecuteGEMV" << std::endl;
    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGemv(device_output, device_input, preloaded_weight);

    FimCopyMemory(o_data, device_output->data, sizeof(half) * num_batch * OUT_LENGTH, DEVICE_TO_HOST);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
}

torch::Tensor py_fim_gemv(torch::Tensor input, torch::Tensor weight, torch::Tensor reorder_t)
{
    int num_batch = input.sizes()[0];
    int num_rows = weight.sizes()[0];
    int num_cols = weight.sizes()[1];

    auto inp_data = torch::flatten(input).data_ptr<at::Half>();
    auto weight_data = torch::flatten(weight).data_ptr<at::Half>();
    int reorder = reorder_t.item<int>();

    std::cout << "Input Num batches : " << num_batch << std::endl;
    std::cout << "Weight Num inputs : " << num_rows << std::endl;
    std::cout << "Weight Num outputs : " << num_cols << std::endl;

    torch::Tensor output = torch::zeros({num_batch, num_cols}, torch::kF16);
    auto out_data = output.data_ptr<at::Half>();

    KernelLauncher(inp_data, weight_data, num_batch, num_rows, num_cols, out_data, reorder);

    return output;
}

static auto registry = torch::RegisterOperators("custom_ops::py_fim_gemv", &py_fim_gemv);
