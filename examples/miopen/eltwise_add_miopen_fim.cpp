#include <miopen/miopen.h>
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <thread>
#include <vector>
#include "fim_runtime_api.h"
#include "utility/fim_dump.hpp"

#include <utility>

int fim_elt_add_miopen()
{
    miopenTensorDescriptor_t a_desc, b_desc, c_desc;

    miopenCreateTensorDescriptor(&a_desc);
    miopenCreateTensorDescriptor(&b_desc);
    miopenCreateTensorDescriptor(&c_desc);

    std::vector<int> a_len = {2, 3, 4, 1};
    std::vector<int> b_len = {2, 3, 4, 1};
    std::vector<int> c_len = {2, 3, 4, 1};

    miopenSetTensorDescriptor(a_desc, miopenHalf, 4, a_len.data(), nullptr);
    miopenSetTensorDescriptor(b_desc, miopenHalf, 4, b_len.data(), nullptr);
    miopenSetTensorDescriptor(c_desc, miopenHalf, 4, c_len.data(), nullptr);

    float alpha_0 = 1;
    float alpha_1 = 1;
    float beta = 1;

    size_t a_sz = a_len[0] * a_len[1] * a_len[2] * a_len[3];
    size_t b_sz = b_len[0] * b_len[1] * b_len[2] * b_len[3];
    size_t c_sz = c_len[0] * c_len[1] * c_len[2] * c_len[3];

    FimBo host_input = {.size = a_sz * sizeof(miopenHalf), .mem_type = MEM_TYPE_HOST};
    FimBo host_weight = {.size = b_sz * sizeof(miopenHalf), .mem_type = MEM_TYPE_HOST};
    FimBo host_output = {.size = c_sz * sizeof(miopenHalf), .mem_type = MEM_TYPE_HOST};
    FimBo device_output = {.size = c_sz * sizeof(miopenHalf), .mem_type = MEM_TYPE_DEVICE};
    FimBo fim_weight = {.size = 2 * b_sz * sizeof(miopenHalf), .mem_type = MEM_TYPE_FIM};

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate host(CPU) memory */
    FimAllocMemory(&host_input);
    FimAllocMemory(&host_weight);
    FimAllocMemory(&host_output);
    /* __FIM_API__ call : Allocate device(GPU) memory */
    FimAllocMemory(&device_output);
    /* __FIM_API__ call : Allocate device(FIM) memory */
    FimAllocMemory(&fim_weight);

    /* Initialize the input, weight, output data */
    load_data("../test_vectors/load/elt_add_input0_64KB.txt", (char*)host_input.data, host_input.size);
    load_data("../test_vectors/load/elt_add_input1_64KB.txt", (char*)host_weight.data, host_weight.size);
    load_data("../test_vectors/load/elt_add_output_64KB.txt", (char*)host_output.data, host_output.size);

    miopenHandle_t handle;
    hipStream_t s;
    hipStreamCreate(&s);
    miopenCreateWithStream(&handle, s);

    miopenOpTensorFIM(handle, miopenTensorOpAdd, &alpha_0, a_desc, &host_input, &alpha_1, b_desc, &host_weight, &beta,
                      c_desc, &fim_weight, &device_output);

    miopenDestroy(handle);

    FimFreeMemory(&fim_weight);
    /* __FIM_API__ call : Free host(CPU) memory */
    FimFreeMemory(&host_input);
    FimFreeMemory(&host_weight);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();
    return 0;
}
int main() { fim_elt_add_miopen(); }
