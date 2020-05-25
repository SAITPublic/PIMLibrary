#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "half.hpp"
#include "utility/fim_dump.hpp"

using half_float::half;

#define LENGTH (128 * 1024)

using namespace std;

int fim_sv_add_1(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_scalar = FimCreateBo(1, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_vector = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_vector = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input0 = test_vector_data + "load/sv_add/scalar_2B.dat";
    std::string input1 = test_vector_data + "load/sv_add/vector_256KB.dat";
    std::string output = test_vector_data + "load/sv_add/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/sv_add/output_256KB.dat";

    load_data(input0.c_str(), (char*)host_scalar->data, host_scalar->size);
    load_data(input1.c_str(), (char*)host_vector->data, host_vector->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_vector, host_vector, HOST_TO_FIM);

    /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
    FimExecuteAdd(device_output, *(half*)host_scalar->data, fim_vector);

    FimCopyMemory(host_output, device_output, FIM_TO_HOST);

    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_scalar);
    FimDestroyBo(host_vector);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_vector);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_sv_mul_1(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_scalar = FimCreateBo(1, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_vector = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* fim_vector = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* device_output = FimCreateBo(LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_FIM);

    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input0 = test_vector_data + "load/sv_mul/scalar_2B.dat";
    std::string input1 = test_vector_data + "load/sv_mul/vector_256KB.dat";
    std::string output = test_vector_data + "load/sv_mul/output_256KB.dat";
    std::string output_dump = test_vector_data + "dump/sv_mul/output_256KB.dat";

    load_data(input0.c_str(), (char*)host_scalar->data, host_scalar->size);
    load_data(input1.c_str(), (char*)host_vector->data, host_vector->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimCopyMemory(fim_vector, host_vector, HOST_TO_FIM);

    /* __FIM_API__ call : Execute FIM kernel (ELT_ADD) */
    FimExecuteMul(device_output, *(half*)host_scalar->data, fim_vector);

    FimCopyMemory(host_output, device_output, FIM_TO_HOST);

    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Free memory */
    FimDestroyBo(host_scalar);
    FimDestroyBo(host_vector);
    FimDestroyBo(host_output);
    FimDestroyBo(golden_output);
    FimDestroyBo(device_output);
    FimDestroyBo(fim_vector);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}
TEST(HIPIntegrationTest, FimSVAdd1) { EXPECT_TRUE(fim_sv_add_1() == 0); }
TEST(HIPIntegrationTest, FimSVMul1) { EXPECT_TRUE(fim_sv_mul_1() == 0); }
