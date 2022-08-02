/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <random>
#include "half.hpp"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_profile.h"

#define IN_LENGTH (256)
#define OUT_LENGTH (4096)
#define BATCH_DIM (2)

#ifdef DEBUG_PIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using half_float::half;
using namespace std;

#define EPSILON (1.0)

class PimGemvTestFixture : public ::testing::Test {
protected:
    virtual void SetUp() override {
        PimInitialize(RT_TYPE_HIP, PIM_FP16);
    }

    virtual void TearDown() override {
        PimDeinitialize();
    }
};


enum GemvDataTypes {INPUT, WEIGHT, OUTPUT, BIAS, GOLDEN};
using GemvFunc = int(*)(PimBo*, PimBo*, PimBo*, void*, bool);

class PimGemvTest {
public:
    PimGemvTest(unsigned n_, unsigned c_, unsigned h_, unsigned w_) : n(n_), c(c_), h(h_), w(w_) {
        // n: batch, c: channel, h: in (height), w: out (width)
        cout << "PimGemvTest: Test start (" << n << ", " << c << ", " << h << ", " << w << ")\n";
        if (n > 1 && c > 1) {
            throw invalid_argument("Both batch and channel size are larger than 1");
        }
        h_i = PimCreateBo(n, c, 1, h, PIM_FP16, MEM_TYPE_HOST);
        h_w = PimCreateBo(n, c, h, w, PIM_FP16, MEM_TYPE_HOST);
        h_o = PimCreateBo(n, c, 1, w, PIM_FP16, MEM_TYPE_HOST);
        d_i = PimCreateBo(n, c, 1, h, PIM_FP16, MEM_TYPE_DEVICE);
        d_w = PimCreateBo(n, c, h, w, PIM_FP16, MEM_TYPE_DEVICE);
        d_o = PimCreateBo(n, c, 1, w, PIM_FP16, MEM_TYPE_DEVICE);
        golden = PimCreateBo(n, c, 1, w, PIM_FP16, MEM_TYPE_HOST);
    }

    void prepare(bool use_random_data=false, float alpha=1.0f, float beta=0.0f) {
        if (use_random_data) {
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(-1.0, 1.0);

            set_half_data((half*)golden->data, half(0.0), (n * c * w));
            set_half_data((half*)h_o->data, half(0.0), (n * c * w));
            if (c == 1) {
                // normal GEMV
                set_half_data((half*)h_i->data, half(dis(gen)), (n * c * h));
                set_half_data((half*)h_w->data, half(dis(gen)), (n * c * h * w));
                matmulCPU((half*)h_i->data, (half*)h_w->data, (half*)golden->data, 1, w, h, half(alpha), half(beta));
            } else if (c > 1) {
                // channel-wise GEMV list
                for (int i = 0; i < c; i++) {
                    set_half_data(((half*)h_i->data) + i * h, half(dis(gen)), h);
                    set_half_data(((half*)h_w->data) + i * h * w, half(dis(gen)), h * w);
                    matmulCPU(((half*)h_i->data) + i * h, ((half*)h_w->data) + i * h * w,
                              ((half*)golden->data) + i * w, 1, w, h, half(alpha), half(beta));
                }
            }
        }
        PimCopyMemory(d_i, h_i, HOST_TO_DEVICE);
        PimCopyMemory(d_w, h_w, HOST_TO_DEVICE);
    }

    void run(GemvFunc pFunc, bool block=true, unsigned niter=10) {
        for (unsigned i = 0; i < niter; ++i) {
            pFunc(d_o, d_i, d_w, nullptr, block);
            if (!block) PimSynchronize();
        }
        PimCopyMemory(h_o, d_o, DEVICE_TO_HOST);
    }

    int validate() {
        return compare_half_relative((half *)golden->data, (half *)h_o->data, (n * c * w), EPSILON);
    }

    void loadDatafromFile(const string file, GemvDataTypes data_type, PimMemType mem_type) {
        string filename = test_vector_data + file;
        FILE* fp = fopen(filename.c_str(), "r");
        if (fp == nullptr) {
            throw invalid_argument("cannot open file");
        }

        auto getPimBo = [&]()-> PimBo* {
            if (data_type == INPUT) {
                return mem_type == MEM_TYPE_HOST ? h_i : d_i;
            }
            if (data_type == WEIGHT) {
                return mem_type == MEM_TYPE_HOST ? h_w : d_w;
            }
            if (data_type == OUTPUT) {
                return mem_type == MEM_TYPE_HOST ? h_o : d_o;
            }
            if (data_type == GOLDEN) {
                return golden;
            }
            throw invalid_argument("invalid type of data");
        };

        PimBo* pimbo = getPimBo();
        char *data = static_cast<char*>(pimbo->data);

        for (size_t i = 0; i < pimbo->size; i++) {
            fscanf(fp, "%c", &data[i]);
        }
        cout << "Loading " << filename << " to PimBo (type: " << data_type << ", mem: " << mem_type << ")\n";
        fclose(fp);
    }

    ~PimGemvTest() {
        cout << "PimGemvTest: Test done" << endl;
        PimDestroyBo(h_i);
        PimDestroyBo(h_w);
        PimDestroyBo(h_o);
        PimDestroyBo(golden);
        PimDestroyBo(d_i);
        PimDestroyBo(d_w);
        PimDestroyBo(d_o);
    }
private:
    unsigned n, c, h, w;

    PimBo* h_i = nullptr;
    PimBo* h_w = nullptr;
    PimBo* h_o = nullptr;
    PimBo* d_i = nullptr;
    PimBo* d_w = nullptr;
    PimBo* d_o = nullptr;
    PimBo* golden = nullptr;

    const string test_vector_data = TEST_VECTORS_DATA;
};

int pim_gemv_batch(bool block)
{
    PimGemvTest t = PimGemvTest(2, 1, 256, 4096);
    t.loadDatafromFile("load/gemv/batch_input_2x256.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/batch_weight_256x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/batch_output_2x4096.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_256(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 256, 4096);
    t.loadDatafromFile("load/gemv/input_256x1.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/weight_256x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/output_4096x1.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_512(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 512, 4096);
    t.loadDatafromFile("load/gemv/input_512x1.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/weight_512x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/output_4096x1_512.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_desc(bool block)
{
    int ret = 0;
    int in_size = 1024;
    int out_size = 4096;
    float alpha = 1.0f;
    float beta = 0.0f;
    //float epsilon = 0.1f; // not-used

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimDesc* pim_desc = PimCreateDesc(1, 1, in_size, out_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    set_half_data((half*)golden_output->data, half(0.0), out_size);
    set_half_data((half*)host_output->data, half(0.0), out_size);
    set_half_data((half*)host_input->data, half(dis(gen)), in_size);
    set_half_data((half*)host_weight->data, half(dis(gen)), in_size * out_size);
    matmulCPU((half*)host_input->data, (half*)host_weight->data, (half*)golden_output->data, 1, out_size, in_size,
              half(alpha), half(beta));

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);
    PimCopyMemory(device_output, host_output, HOST_TO_DEVICE);

    /* __PIM_API__ call : Execute PIM kernel (GEMV) */
    ret = PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
    if (!block) PimSynchronize();

    PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

    ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, EPSILON);

    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_desc_batch(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;
    int batch_n = 4;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimDesc* pim_desc = PimCreateDesc(batch_n, 1, in_size, out_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* temp_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/batch_input_4x1024.dat";
    std::string weight = test_vector_data + "load/gemv/batch_weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/batch_output_4x4096.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/gemv_batch_preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/gemv_batch_output_4x4096.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)temp_output->data, temp_output->size);

    for (int i = 0; i < batch_n; i++) {
        memcpy((half*)golden_output->data + i * pim_desc->bshape_r.w, (half*)temp_output->data + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }

    for (int i = 0; i < pim_desc->bshape_r.w; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.h, (half*)temp_weight->data + i * pim_desc->bshape.h,
               pim_desc->bshape_r.h * sizeof(half));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size * batch_n, EPSILON);
    }
    /* __PIM_API__ call : Destroy PIM Buffer Object */
    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(temp_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_uniform_256(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 256, 4096);
    t.loadDatafromFile("load/gemv/uniform_input_256x1.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/uniform_weight_256x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/uniform_output_4096x1.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_normal_256(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 256, 4096);
    t.loadDatafromFile("load/gemv/normal_input_256x1.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/normal_weight_256x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/normal_output_4096x1.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_uniform_4096(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 4096, 4096);
    t.loadDatafromFile("load/gemv/uniform_input_4096x1.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/uniform_weight_4096x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/uniform_output_4096x4096.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_normal_4096(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 4096, 4096);
    t.loadDatafromFile("load/gemv/normal_input_4096x1.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/normal_weight_4096x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/normal_output_4096x4096.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_no_accum_512(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 512, 4096);
    t.loadDatafromFile("load/gemv/input_512x1.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/weight_512x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/output_4096x1_512.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_no_accum_256(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 256, 4096);
    t.loadDatafromFile("load/gemv/input_256x1.dat", INPUT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/weight_256x4096.dat", WEIGHT, MEM_TYPE_HOST);
    t.loadDatafromFile("load/gemv/output_4096x1_256.dat", GOLDEN, MEM_TYPE_HOST);
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_no_accum_desc(bool block)
{
    int ret = 0;
    int in_size = 800;
    int out_size = 3200;

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    PimExecuteDummy();

    PimDesc* pim_desc = PimCreateDesc(1, 1, in_size, out_size, PIM_FP16, OP_GEMV);
    /* __PIM_API__ call : Create PIM Buffer Object */
    PimBo* host_input = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_INPUT);
    PimBo* host_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* temp_weight = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* host_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* golden_output = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
    PimBo* device_input = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
    PimBo* device_weight = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
    PimBo* device_output = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);

    /* Initialize the input, weight, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string input = test_vector_data + "load/gemv/input_1024x1.dat";
    std::string weight = test_vector_data + "load/gemv/weight_1024x4096.dat";
    std::string output = test_vector_data + "load/gemv/output_4096x1_1024.dat";
    std::string preload_weight = test_vector_data + "dump/gemv/preloaded_weight_1024x4096.dat";
    std::string output_dump = test_vector_data + "dump/gemv/output_4096x1_1024.dat";

    load_data(input.c_str(), (char*)host_input->data, host_input->size);
    load_data(weight.c_str(), (char*)temp_weight->data, temp_weight->size);
    load_data(output.c_str(), (char*)golden_output->data, out_size * sizeof(half));
    for (int i = 0; i < pim_desc->bshape_r.w; i++) {
        memcpy((half*)host_weight->data + i * pim_desc->bshape_r.h, (half*)temp_weight->data + i * pim_desc->bshape.h,
               pim_desc->bshape_r.h * sizeof(half));
    }

    PimCopyMemory(device_input, host_input, HOST_TO_DEVICE);
    PimCopyMemory(device_weight, host_weight, HOST_TO_DEVICE);

    for (int i = 0; i < NUM_ITER; i++) {
        /* __PIM_API__ call : Execute PIM kernel (GEMV) */
        PimExecuteGemv(device_output, device_input, device_weight, nullptr, block);
        if (!block) PimSynchronize();

        PimCopyMemory(host_output, device_output, DEVICE_TO_HOST);

        //    dump_data(preload_weight.c_str(), (char*)preloaded_weight->data, preloaded_weight->size);
        //    dump_data(output_dump.c_str(), (char*)host_output->data, host_output->size);
        ret = compare_half_relative((half*)golden_output->data, (half*)host_output->data, out_size, EPSILON);
    }

    /* __PIM_API__ call : Destroy PIM Buffer Object */

    PimDestroyBo(host_input);
    PimDestroyBo(host_weight);
    PimDestroyBo(temp_weight);
    PimDestroyBo(host_output);
    PimDestroyBo(golden_output);
    PimDestroyBo(device_input);
    PimDestroyBo(device_weight);
    PimDestroyBo(device_output);
    PimDestroyDesc(pim_desc);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

int pim_gemv_moe(bool block)
{
    PimGemvTest t = PimGemvTest(1, 8, 256, 512);
    t.prepare(true, 1.0f, 0.0f);
    t.run(PimExecuteGemvList, true, 10);
    return t.validate();
}

int pim_gemv_moe_chwise(bool block)
{
    PimGemvTest t = PimGemvTest(1, 64, 256, 64);
    t.prepare(true, 1.0f, 0.0f);
    t.run(PimExecuteGemvList, true, 10);
    return t.validate();
}

TEST_F(PimGemvTestFixture, PimGemvBatchSync) { EXPECT_TRUE(pim_gemv_batch(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvBatchAsync) { EXPECT_TRUE(pim_gemv_batch(false) == 0); }
TEST_F(PimGemvTestFixture, PimGemvMoEChwiseSync) { EXPECT_TRUE(pim_gemv_moe_chwise(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvMoESync) { EXPECT_TRUE(pim_gemv_moe(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemv256Sync) { EXPECT_TRUE(pim_gemv_256(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemv256Async) { EXPECT_TRUE(pim_gemv_256(false) == 0); }
TEST_F(PimGemvTestFixture, PimGemv512Sync) { EXPECT_TRUE(pim_gemv_512(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemv512Async) { EXPECT_TRUE(pim_gemv_512(false) == 0); }
TEST(HIPIntegrationTest, PimGemvDescSync) { EXPECT_TRUE(pim_gemv_desc(true) == 0); }
TEST(HIPIntegrationTest, PimGemvDescAsync) { EXPECT_TRUE(pim_gemv_desc(false) == 0); }
TEST(HIPIntegrationTest, PimGemvDescBatchSync) { EXPECT_TRUE(pim_gemv_desc_batch(true) == 0); }
TEST(HIPIntegrationTest, PimGemvDescBatchASync) { EXPECT_TRUE(pim_gemv_desc_batch(false) == 0); }
TEST_F(PimGemvTestFixture, PimGemvUniform256Sync) { EXPECT_TRUE(pim_gemv_uniform_256(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNormal256Sync) { EXPECT_TRUE(pim_gemv_normal_256(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvUniform4096Sync) { EXPECT_TRUE(pim_gemv_uniform_4096(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNormal4096Sync) { EXPECT_TRUE(pim_gemv_normal_4096(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNoAccum512Sync) { EXPECT_TRUE(pim_gemv_no_accum_512(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNoAccum256Sync) { EXPECT_TRUE(pim_gemv_no_accum_256(true) == 0); }
TEST(HIPIntegrationTest, PimGemvNoAccumDescSync) { EXPECT_TRUE(pim_gemv_no_accum_desc(true) == 0); }
