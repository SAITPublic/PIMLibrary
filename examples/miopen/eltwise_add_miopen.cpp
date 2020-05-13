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
#define LENGTH (64 * 1024)

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

    hipMalloc(&a_data, sizeof(half) * LENGTH);
    hipMalloc(&b_data, sizeof(half) * LENGTH);
    hipMalloc(&c_data, sizeof(half) * LENGTH);
    hipMalloc(&ref_data, sizeof(half) * LENGTH);

    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input0 = test_vector_data + "load/elt_add/input0_128KB.dat";
    std::string input1 = test_vector_data + "load/elt_add/input1_128KB.dat";
    std::string output = test_vector_data + "load/elt_add/output_128KB.dat";

    load_data(input0.c_str(), (char *)a_data, sizeof(half) * LENGTH);
    load_data(input1.c_str(), (char *)b_data, sizeof(half) * LENGTH);
    load_data(output.c_str(), (char *)ref_data, sizeof(half) * LENGTH);

    miopenHandle_t handle;
    miopenCreate(&handle);

    miopenOpTensor(handle, miopenTensorOpAdd, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                   c_data);
    miopenDestroy(handle);

    if (compare_data((char *)c_data, (char *)ref_data, sizeof(half) * LENGTH)) ret = -1;

    hipFree(ref_data);
    hipFree(c_data);
    hipFree(b_data);
    hipFree(a_data);

    return ret;
}

TEST(MIOpenIntegrationTest, MIOpenFimEltAdd) { EXPECT_TRUE(miopen_elt_add() == 0); }
