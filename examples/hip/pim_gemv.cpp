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

#ifdef DEBUG_PIM
#define NUM_ITER (100)
#else
#define NUM_ITER (1)
#endif

using half_float::half;
using namespace std;
using GemvFunc = int(*)(PimBo*, PimBo*, PimBo*, void*, bool);

class PimGemvTestFixture : public ::testing::Test {
protected:
    virtual void SetUp() override {
        PimInitialize(RT_TYPE_HIP, PIM_FP16);
    }

    virtual void TearDown() override {
        PimDeinitialize();
    }
};

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

    PimGemvTest(PimDesc* i_desc, PimDesc* w_desc, PimDesc* o_desc) {
        // n: batch, c: channel, h: in (height), w: out (width)
        n = i_desc->bshape_r.n;
        c = i_desc->bshape_r.c;
        h = w_desc->bshape_r.h;
        w = w_desc->bshape_r.w;
        cout << "PimGemvTest: Test start (" << n << ", " << c << ", " << h << ", " << w << ")\n";

        h_i = PimCreateBo(i_desc, MEM_TYPE_HOST, GEMV_INPUT);
        h_w = PimCreateBo(w_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
        h_o = PimCreateBo(o_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
        d_i = PimCreateBo(i_desc, MEM_TYPE_DEVICE, GEMV_INPUT);
        d_w = PimCreateBo(w_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT);
        d_o = PimCreateBo(o_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT);
        golden = PimCreateBo(o_desc, MEM_TYPE_HOST, GEMV_OUTPUT);
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

    int validate(float epsilon=1.0f) {
        return compare_half_relative((half *)golden->data, (half *)h_o->data, (n * c * w), epsilon);
    }

    void loadDatafromFile(PimBo* pimbo, const string file, size_t size=0) {
        string filename = test_vector_data + file;
        FILE* fp = fopen(filename.c_str(), "r");
        if (fp == nullptr) {
            throw invalid_argument("cannot open file");
        }
        char *data = static_cast<char*>(pimbo->data);
        for (size_t i = 0; i < pimbo->size; i++) {
            fscanf(fp, "%c", &data[i]);
        }
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

public:
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
    t.loadDatafromFile(t.h_i, "load/gemv/batch_input_2x256.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/batch_weight_256x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/batch_output_2x4096.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_256(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 256, 4096);
    t.loadDatafromFile(t.h_i, "load/gemv/input_256x1.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/weight_256x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/output_4096x1.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_512(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 512, 4096);
    t.loadDatafromFile(t.h_i, "load/gemv/input_512x1.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/weight_512x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/output_4096x1_512.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_desc(bool block)
{
    PimDesc* pim_desc = PimCreateDesc(1, 1, 1024, 4096, PIM_FP16, OP_GEMV);
    PimGemvTest t = PimGemvTest(pim_desc, pim_desc, pim_desc);
    t.prepare(true);
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_desc_batch(bool block)
{
    PimDesc* pim_desc = PimCreateDesc(4, 1, 800, 3200, PIM_FP16, OP_GEMV);
    PimGemvTest t = PimGemvTest(pim_desc, pim_desc, pim_desc);

    string i_filename(t.test_vector_data + "load/gemv/batch_input_4x1024.dat");
    string w_filename(t.test_vector_data + "load/gemv/batch_weight_1024x4096.dat");
    string o_filename(t.test_vector_data + "load/gemv/batch_output_4x4096.dat");

    PimBo* tmp_w = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* tmp_o = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    load_data(i_filename.c_str(), static_cast<char *>(t.h_i->data), t.h_i->size);
    load_data(w_filename.c_str(), static_cast<char *>(tmp_w->data), tmp_w->size);
    load_data(o_filename.c_str(), static_cast<char *>(tmp_o->data), tmp_o->size);
 
    for (unsigned i = 0; i < pim_desc->bshape.n; i++) {
        memcpy(static_cast<half*>(t.golden->data) + i * pim_desc->bshape_r.w, static_cast<half*>(tmp_o->data) + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }

    for (unsigned i = 0; i < pim_desc->bshape_r.w; i++) {
        memcpy(static_cast<half*>(t.h_w->data) + i * pim_desc->bshape_r.h, static_cast<half*>(tmp_w->data) + i * pim_desc->bshape.h,
               pim_desc->bshape_r.h * sizeof(half));
    }

    t.prepare();
    t.run(PimExecuteGemv, block);

    PimDestroyBo(tmp_o);
    PimDestroyBo(tmp_w);
    return t.validate();
}

int pim_gemv_uniform_256(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 256, 4096);
    t.loadDatafromFile(t.h_i, "load/gemv/uniform_input_256x1.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/uniform_weight_256x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/uniform_output_4096x1.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_normal_256(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 256, 4096);
    t.loadDatafromFile(t.h_i, "load/gemv/normal_input_256x1.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/normal_weight_256x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/normal_output_4096x1.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_uniform_4096(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 4096, 4096);
    t.loadDatafromFile(t.h_i, "load/gemv/uniform_input_4096x1.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/uniform_weight_4096x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/uniform_output_4096x4096.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_normal_4096(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 4096, 4096);
    t.loadDatafromFile(t.h_i, "load/gemv/normal_input_4096x1.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/normal_weight_4096x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/normal_output_4096x4096.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_no_accum_512(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 512, 4096);
    t.loadDatafromFile(t.h_i, "load/gemv/input_512x1.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/weight_512x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/output_4096x1_512.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_no_accum_256(bool block)
{
    PimGemvTest t = PimGemvTest(1, 1, 256, 4096);
    t.loadDatafromFile(t.h_i, "load/gemv/input_256x1.dat");
    t.loadDatafromFile(t.h_w, "load/gemv/weight_256x4096.dat");
    t.loadDatafromFile(t.golden, "load/gemv/output_4096x1_256.dat");
    t.prepare();
    t.run(PimExecuteGemv, block);
    return t.validate();
}

int pim_gemv_no_accum_desc(bool block)
{
    PimDesc* pim_desc = PimCreateDesc(4, 1, 800, 3200, PIM_FP16, OP_GEMV);
    PimGemvTest t = PimGemvTest(pim_desc, pim_desc, pim_desc);

    string i_filename(t.test_vector_data + "load/gemv/input_1024x1.dat");
    string w_filename(t.test_vector_data + "load/gemv/weight_1024x4096.dat");
    string o_filename(t.test_vector_data + "load/gemv/output_4096x1_1024.dat");

    PimBo* tmp_w = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_WEIGHT);
    PimBo* tmp_o = PimCreateBo(pim_desc, MEM_TYPE_HOST, GEMV_OUTPUT);

    load_data(i_filename.c_str(), static_cast<char *>(t.h_i->data), t.h_i->size);
    load_data(w_filename.c_str(), static_cast<char *>(tmp_w->data), tmp_w->size);
    load_data(o_filename.c_str(), static_cast<char *>(tmp_o->data), tmp_o->size);
 
    for (unsigned i = 0; i < pim_desc->bshape.n; i++) {
        memcpy(static_cast<half*>(t.golden->data) + i * pim_desc->bshape_r.w, static_cast<half*>(tmp_o->data) + i * pim_desc->bshape.w,
               pim_desc->bshape_r.w * sizeof(half));
    }

    for (unsigned i = 0; i < pim_desc->bshape_r.w; i++) {
        memcpy(static_cast<half*>(t.h_w->data) + i * pim_desc->bshape_r.h, static_cast<half*>(tmp_w->data) + i * pim_desc->bshape.h,
               pim_desc->bshape_r.h * sizeof(half));
    }

    t.prepare();
    t.run(PimExecuteGemv, block);

    PimDestroyBo(tmp_o);
    PimDestroyBo(tmp_w);
    return t.validate();
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
TEST_F(PimGemvTestFixture, PimGemvDescSync) { EXPECT_TRUE(pim_gemv_desc(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvDescAsync) { EXPECT_TRUE(pim_gemv_desc(false) == 0); }
TEST_F(PimGemvTestFixture, PimGemvDescBatchSync) { EXPECT_TRUE(pim_gemv_desc_batch(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvDescBatchASync) { EXPECT_TRUE(pim_gemv_desc_batch(false) == 0); }
TEST_F(PimGemvTestFixture, PimGemvUniform256Sync) { EXPECT_TRUE(pim_gemv_uniform_256(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNormal256Sync) { EXPECT_TRUE(pim_gemv_normal_256(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvUniform4096Sync) { EXPECT_TRUE(pim_gemv_uniform_4096(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNormal4096Sync) { EXPECT_TRUE(pim_gemv_normal_4096(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNoAccum512Sync) { EXPECT_TRUE(pim_gemv_no_accum_512(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNoAccum256Sync) { EXPECT_TRUE(pim_gemv_no_accum_256(true) == 0); }
TEST_F(PimGemvTestFixture, PimGemvNoAccumDescSync) { EXPECT_TRUE(pim_gemv_no_accum_desc(true) == 0); }
