#include <stdexcept>
#include "fim_runtime_api.h"

#include <torch/script.h>
#include <torch/torch.h>

void KernelLauncher(void* inp0_data, void* inp1_data, int N, int is_scalar, void* out_data, int op)
{
    std::cout << "Launcher for FIM_Eltwise" << std::endl;

    FimDesc* fim_desc = FimCreateDesc(1, 1, 1, N, FIM_FP16);

    // __FIM_API__ call : Create FIM Buffer Object
    FimBo* fim_input1 = FimCreateBo(fim_desc, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_FIM);

    FimCopyMemory((void*)fim_input1->data, (void*)inp1_data, sizeof(uint16_t) * N, HOST_TO_FIM);

    if (is_scalar == 1) {
        uint16_t fim_input0;

        FimCopyMemory((void*)&fim_input0, (void*)inp0_data, sizeof(uint16_t), DEVICE_TO_HOST);

        if (op == 0) {
            std::cout << "Calling FIMExecuteAdd" << std::endl;
            FimExecuteAdd(device_output, (void*)&fim_input0, fim_input1);
        } else {
            std::cout << "Calling FIMExecuteMul" << std::endl;
            FimExecuteMul(device_output, (void*)&fim_input0, fim_input1);
        }
    } else {
        FimBo* fim_input0 = FimCreateBo(fim_desc, MEM_TYPE_FIM);

        FimCopyMemory((void*)fim_input0->data, (void*)inp0_data, sizeof(uint16_t) * N, HOST_TO_FIM);

        if (op == 0) {
            std::cout << "Calling FIMExecuteAdd" << std::endl;
            FimExecuteAdd(device_output, fim_input0, fim_input1);
        } else {
            std::cout << "Calling FIMExecuteMul" << std::endl;
            FimExecuteMul(device_output, fim_input0, fim_input1);
        }

        FimDestroyBo(fim_input0);
    }

    FimCopyMemory((void*)out_data, (void*)device_output->data, sizeof(uint16_t) * N, FIM_TO_HOST);

    //__FIM_API__ call : Free memory
    FimDestroyBo(device_output);
    FimDestroyBo(fim_input1);
    FimDestroyDesc(fim_desc);
}

torch::Tensor py_fim_eltwise(torch::Tensor input0, torch::Tensor input1, torch::Tensor operation)
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

static auto registry = torch::RegisterOperators("custom_ops::py_fim_eltwise", &py_fim_eltwise);
