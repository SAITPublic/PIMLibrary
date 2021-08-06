#include <iostream>
#include "pim_runtime_api.h"
#include "hip/hip_runtime.h"
#include <torch/script.h>
#include <torch/torch.h>
#include "hip/hip_fp16.h"
#include "utility/pim_log.h"

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
}

torch::Tensor py_pim_gemv(torch::Tensor input, torch::Tensor weight, torch::Tensor reorder_t)
{
    int num_batch = input.sizes()[0];
    int num_rows = weight.sizes()[0];
    int num_cols = weight.sizes()[1];

    auto inp_data = torch::flatten(input).data_ptr<at::Half>();
    auto weight_data = torch::flatten(weight).data_ptr<at::Half>();
    int reorder = reorder_t.item<int>();

    DLOG(INFO) << "Input Num batches : " << num_batch << std::endl;
    DLOG(INFO) << "Weight Num inputs : " << num_rows << std::endl;
    DLOG(INFO) << "Weight Num outputs : " << num_cols << std::endl;

    torch::Tensor output = torch::zeros({num_batch, num_cols}, torch::kF16);
    auto out_data = output.data_ptr<at::Half>();

    KernelLauncher(inp_data, weight_data, num_batch, num_rows, num_cols, out_data, reorder);

    return output;
}

static auto registry = torch::RegisterOperators("custom_ops::py_pim_gemv", &py_pim_gemv);
