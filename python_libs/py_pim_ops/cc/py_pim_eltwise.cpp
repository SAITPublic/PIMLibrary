#include <stdexcept>
#include "pim_runtime_api.h"
#include "hip/hip_runtime.h"
#include <torch/script.h>
#include <torch/torch.h>

void KernelLauncher(void* inp0_data, void* inp1_data, int N, int is_scalar, void* out_data, int op)
{
    std::cout << "Launcher for PIM_Eltwise" << std::endl;

    PimDesc* pim_desc = PimCreateDesc(1, 1, 1, N, PIM_FP16);

    // __PIM_API__ call : Create PIM Buffer Object
    PimBo* pim_input1 = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_PIM);

    PimCopyMemory((void*)pim_input1->data, (void*)inp1_data, sizeof(uint16_t) * N, HOST_TO_PIM);

    if (is_scalar == 1) {
        uint16_t pim_input0;

        PimCopyMemory((void*)&pim_input0, (void*)inp0_data, sizeof(uint16_t), DEVICE_TO_HOST);

        if (op == 0) {
            std::cout << "Calling PIMExecuteAdd" << std::endl;
            PimExecuteAdd(device_output, (void*)&pim_input0, pim_input1);
        } else {
            std::cout << "Calling PIMExecuteMul" << std::endl;
            PimExecuteMul(device_output, (void*)&pim_input0, pim_input1);
        }
    } else {
        PimBo* pim_input0 = PimCreateBo(pim_desc, MEM_TYPE_PIM);

        PimCopyMemory((void*)pim_input0->data, (void*)inp0_data, sizeof(uint16_t) * N, HOST_TO_PIM);

        if (op == 0) {
            std::cout << "Calling PIMExecuteAdd" << std::endl;
            PimExecuteAdd(device_output, pim_input0, pim_input1);
        } else {
            std::cout << "Calling PIMExecuteMul" << std::endl;
            PimExecuteMul(device_output, pim_input0, pim_input1);
        }

        PimDestroyBo(pim_input0);
    }

    PimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(uint16_t) * N, PIM_TO_HOST);

    //__PIM_API__ call : Free memory
    PimDestroyBo(device_output);
    PimDestroyBo(pim_input1);
    PimDestroyDesc(pim_desc);
}

torch::Tensor py_pim_eltwise(torch::Tensor input0, torch::Tensor input1, torch::Tensor operation)
{
    // compute num elements in tensor
    int N0 = input0.numel();
    int N1 = input1.numel();
    int op = operation.item<int>();  // 0 for Add and 1 for Mul

    auto inp0_data = torch::flatten(input0).data_ptr<at::Half>();
    auto inp1_data = torch::flatten(input1).data_ptr<at::Half>();

    int wi = -1;  // flag to check which input(wi) tensor is scalar
    int is_scalar = 0;
    torch::Tensor output;

    if (N0 != N1) {
        // if num elems are not equal then check if one of them is a scalar.
        if (N0 != 1 && N1 != 1) {
            throw std::runtime_error("num elems in both tensors are not same and neither of them is a scalar\n");
        } else if (N0 == 1)
            wi = 0;  // input0 is scalar
        else
            wi = 1;  // input1 is scalar
    }

    if (wi != -1) is_scalar = 1;  // this means one of the inp tensors is a scalar

    if (wi == 1) {
        output = torch::zeros_like(input0);
        auto out_data = output.data_ptr<at::Half>();
        KernelLauncher(inp1_data, inp0_data, N0, is_scalar, out_data, op);

    } else {
        output = torch::zeros_like(input1);
        auto out_data = output.data_ptr<at::Half>();
        KernelLauncher(inp0_data, inp1_data, N1, is_scalar, out_data, op);
    }

    return output;
}

static auto registry = torch::RegisterOperators("custom_ops::py_pim_eltwise", &py_pim_eltwise);
