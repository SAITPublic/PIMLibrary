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

int miopen_elt_mul()
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
    std::string input0 = test_vector_data + "load/elt_mul/input0_256KB.dat";
    std::string input1 = test_vector_data + "load/elt_mul/input1_256KB.dat";
    std::string output = test_vector_data + "load/elt_mul/output_256KB.dat";

    load_data(input0.c_str(), (char *)a_data, sizeof(miopenHalf) * LENGTH);
    load_data(input1.c_str(), (char *)b_data, sizeof(miopenHalf) * LENGTH);
    load_data(output.c_str(), (char *)ref_data, sizeof(miopenHalf) * LENGTH);

    miopenHandle_t handle;
    hipStream_t stream;

    hipStreamCreate(&stream);
    miopenCreateWithStream(&handle, stream);

    // For profiling, add extra call for GPU initialization
    miopenOpTensor(handle, miopenTensorOpMul, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                   c_data);

    for (int iter = 0; iter < 1; iter++) {
        miopenOpTensor(handle, miopenTensorOpMul, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
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

TEST(MIOpenIntegrationTest, MIOpenFimEltMul) { EXPECT_TRUE(miopen_elt_mul() == 0); }
