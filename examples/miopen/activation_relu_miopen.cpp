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
#define LENGTH (128 * 1024)

int miopen_relu()
{
    int ret = 0;
    void *a_data, *b_data, *ref_data;
    miopenTensorDescriptor_t a_desc, b_desc;
    miopenActivationDescriptor_t activDesc;

    miopenCreateTensorDescriptor(&a_desc);
    miopenCreateTensorDescriptor(&b_desc);

    std::vector<int> a_len = {LENGTH};
    std::vector<int> b_len = {LENGTH};

    miopenSetTensorDescriptor(a_desc, miopenHalf, 1, a_len.data(), nullptr);
    miopenSetTensorDescriptor(b_desc, miopenHalf, 1, b_len.data(), nullptr);

    hipMalloc(&a_data, sizeof(half) * LENGTH);
    hipMalloc(&b_data, sizeof(half) * LENGTH);
    hipMalloc(&ref_data, sizeof(half) * LENGTH);

    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input = test_vector_data + "load/relu/input_256KB.dat";
    std::string output = test_vector_data + "load/relu/output_256KB.dat";

    load_data(input.c_str(), (char *)a_data, sizeof(half) * LENGTH);
    load_data(output.c_str(), (char *)ref_data, sizeof(half) * LENGTH);

    miopenHandle_t handle;
    miopenCreate(&handle);

    miopenCreateActivationDescriptor(&activDesc);

    float alpha = 1, beta = 0;
    miopenSetActivationDescriptor(activDesc, miopenActivationRELU, 1, 0, 0);
    miopenActivationForward(handle, activDesc, &alpha, a_desc, a_data, &beta, b_desc, b_data);

    miopenDestroy(handle);
    miopenDestroyActivationDescriptor(activDesc);

    if (compare_data((char *)b_data, (char *)ref_data, sizeof(half) * LENGTH)) ret = -1;

    hipFree(ref_data);
    hipFree(b_data);
    hipFree(a_data);

    return ret;
}

TEST(MIOpenIntegrationTest, MIOpenRelu) { EXPECT_TRUE(miopen_relu() == 0); }
