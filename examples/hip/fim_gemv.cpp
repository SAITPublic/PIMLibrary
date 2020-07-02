#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fim_runtime_api.h"
#include "hip/hip_fp16.h"
#include "utility/fim_dump.hpp"

#define IN_LENGTH (256)
#define OUT_LENGTH (4096)
#define BATCH_DIM (2)

using namespace std;

int fim_gemv_batch(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(IN_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_reordered_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_input = FimCreateBo(IN_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* device_output = FimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* host_output = FimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(OUT_LENGTH, 1, 1, BATCH_DIM, FIM_FP16, MEM_TYPE_HOST);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input = test_vector_data + "load/gemv/batch_input_2x256.dat";
    std::string weight = test_vector_data + "load/gemv/batch_weight_256x4096.dat";
    std::string output = test_vector_data + "load/gemv/batch_output_2x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/batch_preloaded_weight_256x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/batch_output_2x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);

    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGEMV(device_output, device_input, preloaded_weight);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(IN_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_reordered_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_input = FimCreateBo(IN_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* device_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_weight = FimCreateBo(IN_LENGTH, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* host_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input = test_vector_data + "load/gemv/input_256x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_256x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_256x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGEMV(device_output, device_input, preloaded_weight);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);
    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);

    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv2(void)
{
    int ret = 0;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(IN_LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_weight = FimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* host_reordered_weight = FimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* device_input = FimCreateBo(IN_LENGTH * 2, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* device_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_DEVICE);
    FimBo* preloaded_weight = FimCreateBo(IN_LENGTH * 2, OUT_LENGTH, 1, 1, FIM_FP16, MEM_TYPE_FIM);
    FimBo* host_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);
    FimBo* golden_output = FimCreateBo(OUT_LENGTH, 1, 1, 1, FIM_FP16, MEM_TYPE_HOST);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input = test_vector_data + "load/gemv/input_512x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_512x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1_512.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_512x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1_512.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)host_weight->data, host_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, golden_output->size);

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_DEVICE);

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGEMV(device_output, device_input, preloaded_weight);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
    ret = compare_data((char*)golden_output->data, (char*)host_output->data, host_output->size);

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv3(void)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimExecuteDummy();

    FimDesc* fim_desc = FimCreateDesc(1, 1, out_size, in_size, FIM_FP16, OP_GEMV);
    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    FimBo* host_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* temp_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* host_reordered_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* device_input = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    FimBo* preloaded_weight = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    FimBo* golden_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input = test_vector_data + "load/gemv/gemv_input_1024x4096.dat";
    std::string weight = test_vector_data + "load/gemv/gemv_weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/gemv_output_1024x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/gemv_preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/gemv_output_1024x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, out_size * sizeof(half));

    for (int i = 0; i < fim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * fim_desc->bshape_r.w, (half*)temp_weight->data + i * fim_desc->bshape.w,
               fim_desc->bshape_r.w * sizeof(half));
    }

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_FIM);

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGEMV(device_output, device_input, preloaded_weight);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
    ret = compare_data((char*)golden_output->data, (char*)host_output->data, out_size * sizeof(half));

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(temp_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(golden_output);
    FimDestroyDesc(fim_desc);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

int fim_gemv4(void)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;
    int batch_n = 4;

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);

    FimDesc* fim_desc = FimCreateDesc(batch_n, 1, out_size, in_size, FIM_FP16, OP_GEMV);
    /* __FIM_API__ call : Create FIM Buffer Object */
    FimBo* host_input = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    FimBo* host_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* temp_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* host_reordered_weight = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    FimBo* device_input = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    FimBo* device_output = FimCreateBo(fim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
    FimBo* preloaded_weight = FimCreateBo(fim_desc, MEM_TYPE_FIM, GEMV_WEIGHT);
    FimBo* host_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    FimBo* temp_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    FimBo* golden_output = FimCreateBo(fim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input = test_vector_data + "load/gemv/gemv_batch_input_4x1024.dat";
    std::string weight = test_vector_data + "load/gemv/gemv_batch_weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/gemv_batch_output_4x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/gemv_batch_preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/gemv_batch_output_4x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)temp_output->data, temp_output->size);

    for (int i = 0; i < batch_n; i++) {
        memcpy((half*)golden_output->data + i * fim_desc->bshape_r.h, (half*)temp_output->data + i * fim_desc->bshape.h,
               fim_desc->bshape_r.h * sizeof(half));
    }

    for (int i = 0; i < fim_desc->bshape_r.h; i++) {
        memcpy((half*)host_weight->data + i * fim_desc->bshape_r.w, (half*)temp_weight->data + i * fim_desc->bshape.w,
               fim_desc->bshape_r.w * sizeof(half));
    }

    FimCopyMemory(device_input, host_input, HOST_TO_DEVICE);

    /* __FIM_API__ call : Preload weight data on FIM memory */
    FimConvertDataLayout(host_reordered_weight, host_weight, OP_GEMV);
    FimCopyMemory(preloaded_weight, host_reordered_weight, HOST_TO_FIM);

    /* __FIM_API__ call : Execute FIM kernel (GEMV) */
    FimExecuteGEMV(device_output, device_input, preloaded_weight);

    FimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
    ret = compare_data((char*)golden_output->data, (char*)host_output->data, batch_n * out_size * sizeof(half));

    /* __FIM_API__ call : Destroy FIM Buffer Object */
    FimDestroyBo(host_input);
    FimDestroyBo(host_weight);
    FimDestroyBo(temp_weight);
    FimDestroyBo(host_output);
    FimDestroyBo(device_input);
    FimDestroyBo(device_output);
    FimDestroyBo(preloaded_weight);
    FimDestroyBo(host_reordered_weight);
    FimDestroyBo(temp_output);
    FimDestroyBo(golden_output);
    FimDestroyDesc(fim_desc);

    /* __FIM_API__ call : Deinitialize FimRuntime */
    FimDeinitialize();

    return ret;
}

TEST(HIPIntegrationTest, FimGEMVBATCH) { EXPECT_TRUE(fim_gemv_batch() == 0); }
TEST(HIPIntegrationTest, FimGEMV) { EXPECT_TRUE(fim_gemv() == 0); }
TEST(HIPIntegrationTest, FimGEMV2) { EXPECT_TRUE(fim_gemv2() == 0); }
TEST(HIPIntegrationTest, FimGEMV3) { EXPECT_TRUE(fim_gemv3() == 0); }
TEST(HIPIntegrationTest, FimGEMV4) { EXPECT_TRUE(fim_gemv4() == 0); }
