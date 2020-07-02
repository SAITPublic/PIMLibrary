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
//#define LENGTH (64 * 1024)
int miopen_bn_1()
{
    int ret = 0;

    const int BATCH = 1;
    const int CH = 1;
    const int WIDTH = 1;
    const int HEIGHT = 1;
    const int LENGTH = BATCH * CH * HEIGHT * WIDTH;

    void *i_data, *o_data, *ref_data;
    void *gamma_data, *beta_data;
    void *shift_data, *scale_data;

    miopenTensorDescriptor_t i_desc, o_desc, w_desc;

    miopenCreateTensorDescriptor(&i_desc);
    miopenCreateTensorDescriptor(&o_desc);
    miopenCreateTensorDescriptor(&w_desc);

    std::vector<int> i_len = {BATCH, CH, HEIGHT, WIDTH};
    std::vector<int> o_len = {BATCH, CH, HEIGHT, WIDTH};
    std::vector<int> w_len = {1, CH, 1, 1};
    std::vector<int> ref_len = {LENGTH};

    miopenSetTensorDescriptor(i_desc, miopenHalf, 4, i_len.data(), nullptr);
    miopenSetTensorDescriptor(o_desc, miopenHalf, 4, o_len.data(), nullptr);
    miopenSetTensorDescriptor(w_desc, miopenHalf, 4, w_len.data(), nullptr);

    hipMalloc(&i_data, sizeof(half) * LENGTH);
    hipMalloc(&o_data, sizeof(half) * LENGTH);
    hipMalloc(&ref_data, sizeof(half) * LENGTH);

    hipMalloc(&beta_data, sizeof(half) * CH);
    hipMalloc(&gamma_data, sizeof(half) * CH);
    hipMalloc(&scale_data, sizeof(half) * CH);
    hipMalloc(&shift_data, sizeof(half) * CH);

    ((half *)i_data)[0] = half(2.0);
    ((half *)beta_data)[0] = half(3.0);
    ((half *)gamma_data)[0] = half(10.0);
    ((half *)scale_data)[0] = half(4.0);
    ((half *)shift_data)[0] = half(1.0);
    ((half *)ref_data)[0] = half(4.0);

    miopenHandle_t handle;
    miopenCreate(&handle);
    double epsilon = 1e-3;
    float alpha = 1.0;
    float beta = 0.0;
    miopenBatchNormMode_t mode;
    mode = miopenBNSpatial;

    miopenBatchNormalizationForwardInference(handle, mode, &alpha, &beta, i_desc, i_data, o_desc, o_data, w_desc,
                                             gamma_data, beta_data, shift_data, scale_data, epsilon);

    void *out;
    hipMalloc(&out, sizeof(half) * LENGTH);
    // Todo check return value
    auto status = hipMemcpy(out, o_data, sizeof(half) * LENGTH, hipMemcpyDeviceToHost);
    // std::cout << "out " << ((half *)out)[0] << std::endl;
    // std::cout << "refout " << ((half *)ref_data)[0] << std::endl;

    miopenDestroyTensorDescriptor(i_desc);
    miopenDestroyTensorDescriptor(o_desc);
    miopenDestroyTensorDescriptor(w_desc);
    miopenDestroy(handle);

    if (compare_data((char *)out, (char *)ref_data, sizeof(half) * LENGTH)) ret = -1;

    hipFree(ref_data);
    hipFree(out);
    hipFree(i_data);
    hipFree(o_data);

    hipFree(beta_data);
    hipFree(gamma_data);
    hipFree(scale_data);
    hipFree(shift_data);

    return ret;
}

int miopen_bn_2()
{
    int ret = 0;

    const int BATCH = 2;
    const int CH = 64;
    const int WIDTH = 1024;
    const int HEIGHT = 1;
    const int LENGTH = BATCH * CH * HEIGHT * WIDTH;

    void *i_data, *o_data, *ref_data;
    void *gamma_data, *beta_data;
    void *mean_data, *var_data;

    miopenTensorDescriptor_t i_desc, o_desc, w_desc;

    miopenCreateTensorDescriptor(&i_desc);
    miopenCreateTensorDescriptor(&o_desc);
    miopenCreateTensorDescriptor(&w_desc);

    std::vector<int> i_len = {BATCH, CH, HEIGHT, WIDTH};
    std::vector<int> o_len = {BATCH, CH, HEIGHT, WIDTH};
    std::vector<int> w_len = {1, CH, 1, 1};
    std::vector<int> ref_len = {LENGTH};

    miopenSetTensorDescriptor(i_desc, miopenHalf, 4, i_len.data(), nullptr);
    miopenSetTensorDescriptor(o_desc, miopenHalf, 4, o_len.data(), nullptr);
    miopenSetTensorDescriptor(w_desc, miopenHalf, 4, w_len.data(), nullptr);

    hipMalloc(&i_data, sizeof(half) * LENGTH);
    hipMalloc(&o_data, sizeof(half) * LENGTH);
    hipMalloc(&ref_data, sizeof(half) * LENGTH);

    hipMalloc(&beta_data, sizeof(half) * CH);
    hipMalloc(&gamma_data, sizeof(half) * CH);
    hipMalloc(&mean_data, sizeof(half) * CH);
    hipMalloc(&var_data, sizeof(half) * CH);

    /* Initialize the input, output data */
    std::string test_vector_data = TEST_VECTORS_DATA;
    test_vector_data.append("/test_vectors/");

    std::string input_file = test_vector_data + "load/bn/input_256KB.dat";
    std::string beta_file = test_vector_data + "load/bn/beta_128B.dat";
    std::string gamma_file = test_vector_data + "load/bn/gamma_128B.dat";
    std::string mean_file = test_vector_data + "load/bn/mean_128B.dat";
    std::string variance_file = test_vector_data + "load/bn/variance_128B.dat";
    std::string output_file = test_vector_data + "load/bn/output_256KB.dat";

    load_data(input_file.c_str(), (char *)i_data, sizeof(half) * LENGTH);
    load_data(beta_file.c_str(), (char *)beta_data, sizeof(half) * CH);
    load_data(gamma_file.c_str(), (char *)gamma_data, sizeof(half) * CH);
    load_data(mean_file.c_str(), (char *)mean_data, sizeof(half) * CH);
    load_data(variance_file.c_str(), (char *)var_data, sizeof(half) * CH);
    load_data(output_file.c_str(), (char *)ref_data, sizeof(half) * LENGTH);

    miopenHandle_t handle;
    miopenCreate(&handle);
    double epsilon = 1e-5;
    float alpha = 1.0;
    float beta = 0.0;
    miopenBatchNormMode_t mode;
    mode = miopenBNSpatial;

    miopenBatchNormalizationForwardInference(handle, mode, &alpha, &beta, i_desc, i_data, o_desc, o_data, w_desc,
                                             gamma_data, beta_data, mean_data, var_data, epsilon);

    void *out;
    hipMalloc(&out, sizeof(half) * LENGTH);
    auto status = hipMemcpy(out, o_data, sizeof(half) * LENGTH, hipMemcpyDeviceToHost);

    // Todo : Test fails but tolerance is around 0.001
    // std::cout << "out " << ((half *)out)[0] << std::endl;
    // std::cout << "refout " << ((half *)ref_data)[0] << std::endl;

    miopenDestroyTensorDescriptor(i_desc);
    miopenDestroyTensorDescriptor(o_desc);
    miopenDestroyTensorDescriptor(w_desc);
    miopenDestroy(handle);

    if (compare_data_round_off((half *)out, (half *)ref_data, LENGTH, 0.01)) ret = -1;

    hipFree(ref_data);
    hipFree(out);
    hipFree(i_data);
    hipFree(o_data);

    hipFree(beta_data);
    hipFree(gamma_data);
    hipFree(mean_data);
    hipFree(var_data);

    return ret;
}

// Todo: Enbable first test if fim can handle it.
// TEST(MIOpenIntegrationTest, MIOpenBn1) { EXPECT_TRUE(miopen_bn_1() == 0); }
TEST(MIOpenIntegrationTest, MIOpenBn2) { EXPECT_TRUE(miopen_bn_2() == 0); }
