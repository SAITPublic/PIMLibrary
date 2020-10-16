#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <stdlib.h>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "utility/fim_dump.hpp"
#include "utility/fim_profile.h"
#define LENGTH (128 * 1024)

int miopen_elt_add()
{
    int ret = 0;
    void *a_data, *b_data, *c_data, *ref_data;
    miopenTensorDescriptor_t a_desc, b_desc, c_desc;

    miopenCreateTensorDescriptor(&a_desc);
    miopenCreateTensorDescriptor(&b_desc);
    miopenCreateTensorDescriptor(&c_desc);

    std::vector<int> a_len = {LENGTH};
    std::vector<int> b_len = {LENGTH};
    std::vector<int> c_len = {LENGTH};
    std::vector<int> ref_len = {LENGTH};

    miopenSetTensorDescriptor(a_desc, miopenHalf, 1, a_len.data(), nullptr);
    miopenSetTensorDescriptor(b_desc, miopenHalf, 1, b_len.data(), nullptr);
    miopenSetTensorDescriptor(c_desc, miopenHalf, 1, c_len.data(), nullptr);

    float alpha_0 = 1;
    float alpha_1 = 1;
    float beta = 1;

    hipHostMalloc(&a_data, sizeof(miopenHalf) * LENGTH);
    hipHostMalloc(&b_data, sizeof(miopenHalf) * LENGTH);
    hipHostMalloc(&c_data, sizeof(miopenHalf) * LENGTH);
    hipHostMalloc(&ref_data, sizeof(miopenHalf) * LENGTH);

    std::string test_vector_data = TEST_VECTORS_DATA;
    std::string input0 = test_vector_data + "load/elt_add/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_256KB.dat";

    load_data(input0.c_str(), (char *)a_data, sizeof(miopenHalf) * LENGTH);
    load_data(input1.c_str(), (char *)b_data, sizeof(miopenHalf) * LENGTH);
    load_data(output.c_str(), (char *)ref_data, sizeof(miopenHalf) * LENGTH);

    miopenHandle_t handle;
    miopenCreate(&handle);
    hipStream_t stream;

    hipStreamCreate(&stream);
    miopenCreateWithStream(&handle, stream);

    miopenOpTensor(handle, miopenTensorOpAdd, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                   c_data);
    for (int iter = 0; iter < 1000; iter++) {
        miopenOpTensor(handle, miopenTensorOpAdd, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                       c_data);
    }
    hipStreamSynchronize(stream);
    miopenDestroy(handle);
    hipHostFree(ref_data);
    hipHostFree(c_data);
    hipHostFree(b_data);
    hipHostFree(a_data);

    return ret;
}

#include "hip/hip_runtime.h"
int miopen_elt_add_batch_profile(int batch)
{
    int ret = 0;
    void *a_data, *b_data, *c_data, *ref_data;
    miopenTensorDescriptor_t a_desc, b_desc, c_desc;

    miopenCreateTensorDescriptor(&a_desc);
    miopenCreateTensorDescriptor(&b_desc);
    miopenCreateTensorDescriptor(&c_desc);

    std::vector<int> a_len = {batch * LENGTH};
    std::vector<int> b_len = {batch * LENGTH};
    std::vector<int> c_len = {batch * LENGTH};
    std::vector<int> ref_len = {batch * LENGTH};

    miopenSetTensorDescriptor(a_desc, miopenHalf, 1, a_len.data(), nullptr);
    miopenSetTensorDescriptor(b_desc, miopenHalf, 1, b_len.data(), nullptr);
    miopenSetTensorDescriptor(c_desc, miopenHalf, 1, c_len.data(), nullptr);

    float alpha_0 = 1;
    float alpha_1 = 1;
    float beta = 1;

    hipHostMalloc(&a_data, sizeof(miopenHalf) * batch * LENGTH);
    hipHostMalloc(&b_data, sizeof(miopenHalf) * batch * LENGTH);
    hipHostMalloc(&c_data, sizeof(miopenHalf) * batch * LENGTH);
    hipHostMalloc(&ref_data, sizeof(miopenHalf) * batch * LENGTH);

    miopenHandle_t handle;
    hipStream_t stream;

    hipStreamCreate(&stream);
    miopenCreateWithStream(&handle, stream);

    miopenOpTensor(handle, miopenTensorOpAdd, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                   c_data);
    for (int iter = 0; iter < 1; iter++) {
        miopenOpTensor(handle, miopenTensorOpAdd, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                       c_data);
    }

    hipStreamSynchronize(stream);
    miopenDestroy(handle);

    hipHostFree(ref_data);
    hipHostFree(c_data);
    hipHostFree(b_data);
    hipHostFree(a_data);

    return ret;
}

TEST(MIOpenIntegrationTest, MIOpenFimEltAdd) { EXPECT_TRUE(miopen_elt_add() == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile1) { EXPECT_TRUE(miopen_elt_add_batch_profile(1) == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile2) { EXPECT_TRUE(miopen_elt_add_batch_profile(2) == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile4) { EXPECT_TRUE(miopen_elt_add_batch_profile(4) == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile8) { EXPECT_TRUE(miopen_elt_add_batch_profile(8) == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile16) { EXPECT_TRUE(miopen_elt_add_batch_profile(16) == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile32) { EXPECT_TRUE(miopen_elt_add_batch_profile(32) == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile64) { EXPECT_TRUE(miopen_elt_add_batch_profile(64) == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile128) { EXPECT_TRUE(miopen_elt_add_batch_profile(128) == 0); }
TEST(MIOpenIntegrationTest, MIOpenFimEltAddBatchProfile256) { EXPECT_TRUE(miopen_elt_add_batch_profile(256) == 0); }
