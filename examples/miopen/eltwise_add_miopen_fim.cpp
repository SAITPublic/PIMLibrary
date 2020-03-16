#include <miopen/miopen.h>
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <thread>
#include <vector>
#include "fim_runtime_api.h"
#include "half.hpp"
#include "utility/fim_dump.hpp"

using half_float::half;
#include <utility>

#define LENGTH (64 * 1024)

int fim_elt_add_miopen()
{
    miopenTensorDescriptor_t a_desc, b_desc, c_desc;

    miopenCreateTensorDescriptor(&a_desc);
    miopenCreateTensorDescriptor(&b_desc);
    miopenCreateTensorDescriptor(&c_desc);

    std::vector<int> a_len = {LENGTH};
    std::vector<int> b_len = {LENGTH};
    std::vector<int> c_len = {LENGTH};

    miopenSetTensorDescriptor(a_desc, miopenHalf, 4, a_len.data(), nullptr);
    miopenSetTensorDescriptor(b_desc, miopenHalf, 4, b_len.data(), nullptr);
    miopenSetTensorDescriptor(c_desc, miopenHalf, 4, c_len.data(), nullptr);

    float alpha_0 = 1;
    float alpha_1 = 1;
    float beta = 0;

    FimBo host_input0 = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo host_input1 = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo host_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_HOST};
    FimBo device_output = {.size = LENGTH * sizeof(half), .mem_type = MEM_TYPE_DEVICE};

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Allocate host(CPU) memory */
    FimAllocMemory(&host_input0);
    FimAllocMemory(&host_input1);
    FimAllocMemory(&host_output);
    /* __FIM_API__ call : Allocate device(GPU) memory */
    FimAllocMemory(&device_output);

    /* Initialize the input, weight, output data */
    load_data("../test_vectors/load/elt_add/input0_128KB.dat", (char*)host_input0.data, host_input0.size);
    load_data("../test_vectors/load/elt_add/input1_128KB.dat", (char*)host_input1.data, host_input1.size);
    load_data("../test_vectors/load/elt_add/output_128KB.dat", (char*)host_output.data, host_output.size);

    miopenHandle_t handle;
    hipStream_t s;
    hipStreamCreate(&s);
    miopenCreateWithStream(&handle, s);

    miopenOpTensor(handle, miopenTensorOpAdd, &alpha_0, a_desc, (void*)&host_input0, &alpha_1, b_desc,
                   (void*)&host_input1, &beta, c_desc, (void*)&device_output);

    miopenDestroy(handle);

    /* __FIM_API__ call : Free host(CPU) memory */
    FimFreeMemory(&host_input0);
    FimFreeMemory(&host_input1);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();
    return 0;
}
int main() { fim_elt_add_miopen(); }
