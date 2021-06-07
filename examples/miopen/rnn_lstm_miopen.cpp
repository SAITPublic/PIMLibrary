/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <stdlib.h>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "half.hpp"
#include "pim_runtime_api.h"
#include "utility/pim_dump.hpp"
#define LENGTH (64 * 1024)

using half_float::half;

inline float convertH2F(half h_val) { return half_float::detail::half2float<float>(h_val); }
inline int compare_data_round_off(half *data_a, half *data_b, size_t size, double epsilon = 0.001)
{
    int pass_cnt = 0;
    int fail_cnt = 0;
    int ret = 0;
    float abs_diff;
    float max_diff = 0.0;
    float avg_diff = 0.0;

    for (int i = 0; i < size; i++) {
        abs_diff = abs(float(data_a[i]) - float(data_b[i]));
        if (abs_diff < epsilon) {
            pass_cnt++;
        } else {
            fail_cnt++;
            if (max_diff < abs_diff) max_diff = abs_diff;
            avg_diff += abs_diff;
            ret = 1;
        }
    }
    avg_diff /= fail_cnt;

    if (ret) {
        printf("pass_cnt : %d, fail_cnt : %d, pass ratio : %f\n", pass_cnt, fail_cnt,
               ((float)pass_cnt / ((float)fail_cnt + (float)pass_cnt) * 100.));
        printf("max_diff : %f, avg_diff : %f\n", max_diff, avg_diff);
    }

    return ret;
}

int miopen_rnn_lstm()
{
    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);

    void *in_dev, *hx_dev, *out_dev, *wei_dev, *cx_dev, *workspace_dev, *hy_dev, *cy_dev;
    miopenTensorDescriptor_t input_tensor, hidden_tensor, weight_tensor, output_tensor;
    std::vector<miopenTensorDescriptor_t> input_tensors;
    std::vector<miopenTensorDescriptor_t> output_tensors;
    miopenHandle_t handle;
    miopenRNNDescriptor_t rnnDesc;

    int ret = 0;
    miopenCreate(&handle);

    miopenCreateTensorDescriptor(&input_tensor);
    miopenCreateTensorDescriptor(&hidden_tensor);
    miopenCreateTensorDescriptor(&weight_tensor);
    miopenCreateTensorDescriptor(&output_tensor);
    miopenCreateRNNDescriptor(&rnnDesc);

    int nseq = 1;                  // Number of iterations to unroll over
    int in_h = 1024;               // Input Length
    std::vector<int> in_len({1});  // input tensor length
    in_len.push_back(in_h);

    int hid_h = 512;                           // Hidden State Length
    int hid_l = 1 * 2;                         // Number of hidden stacks, *2 for bidirection
    std::vector<int> hid_len({hid_l, hid_h});  // hidden tensor length

    int out_h = hid_h * 2;              // for bidirection
    std::vector<int> out_len({out_h});  // output tensor length

    for (int i = 0; i < in_len.size() - 1; i++) {
        std::array<int, 2> in_lens = {{in_len[i], in_len.back()}};
        miopenCreateTensorDescriptor(&input_tensor);
        miopenSetTensorDescriptor(input_tensor, miopenHalf, 2, in_lens.data(), nullptr);
        input_tensors.push_back(input_tensor);

        std::array<int, 2> out_lens = {{in_len[i], out_len[0]}};
        miopenCreateTensorDescriptor(&output_tensor);
        miopenSetTensorDescriptor(output_tensor, miopenHalf, 2, out_lens.data(), nullptr);
        output_tensors.push_back(output_tensor);
    }

    std::array<int, 3> hid_lens = {{hid_len[0], in_len[0], hid_len[1]}};
    miopenSetTensorDescriptor(hidden_tensor, miopenHalf, 3, hid_lens.data(), nullptr);

    int layer = 1;     // Number of hidden stacks
    int wei_hh = 512;  // Hidden state length
    miopenRNNMode_t mode = miopenLSTM;
    miopenRNNBiasMode_t biasMode = miopenRNNNoBias;
    miopenRNNDirectionMode_t directionMode;
    directionMode = miopenRNNbidirection;
    miopenRNNInputMode_t inMode = miopenRNNlinear;
    miopenRNNAlgo_t algo = miopenRNNdefault;

    miopenSetRNNDescriptor(rnnDesc, wei_hh, layer, inMode, directionMode, mode, biasMode, algo, miopenHalf);
    miopenGetRNNParamsDescriptor(handle, rnnDesc, input_tensor, weight_tensor, miopenHalf);

    size_t in_sz = 0;
    size_t out_sz = 0;
    size_t wei_sz = 0;
    size_t hy_sz = 0;
    size_t workSpace_sz;

    miopenGetRNNInputTensorSize(handle, rnnDesc, nseq, input_tensors.data(), &in_sz);
    miopenGetRNNInputTensorSize(handle, rnnDesc, nseq, output_tensors.data(), &out_sz);
    miopenGetRNNHiddenTensorSize(handle, rnnDesc, nseq, input_tensors.data(), &hy_sz);
    miopenGetRNNWorkspaceSize(handle, rnnDesc, nseq, input_tensors.data(), &workSpace_sz);
    miopenGetRNNParamsSize(handle, rnnDesc, input_tensors[0], &wei_sz, miopenHalf);

    in_sz /= sizeof(half);
    out_sz /= sizeof(half);
    hy_sz /= sizeof(half);
    wei_sz /= sizeof(half);
    workSpace_sz /= sizeof(half);

    hipHostMalloc(&in_dev, sizeof(half) * in_sz);
    hipHostMalloc(&hx_dev, sizeof(half) * hy_sz);
    hipHostMalloc(&out_dev, sizeof(half) * out_sz);
    hipHostMalloc(&wei_dev, sizeof(half) * wei_sz);
    hipHostMalloc(&cx_dev, sizeof(half) * hy_sz);
    hipHostMalloc(&workspace_dev, sizeof(half) * workSpace_sz);
    hipHostMalloc(&hy_dev, sizeof(half) * hy_sz);
    hipHostMalloc(&cy_dev, sizeof(half) * hy_sz);

    /******************************
    Initialization begin
    *****************************/

    std::vector<half> in;
    std::vector<half> wei;
    std::vector<half> out;
    std::vector<half> hx;
    std::vector<half> cx;
    std::vector<half> hy;
    std::vector<half> cy;
    std::vector<half> workspace;
    std::vector<double> outhost;
    std::vector<double> hy_host;
    std::vector<double> cy_host;

    in = std::vector<half>(in_sz);
    hx = std::vector<half>(hy_sz, static_cast<half>(0));
    wei = std::vector<half>(wei_sz);
    out = std::vector<half>(out_sz, static_cast<half>(0));
    cx = std::vector<half>(hy_sz, static_cast<half>(0));
    hy = std::vector<half>(hy_sz, static_cast<half>(0));
    cy = std::vector<half>(hy_sz, static_cast<half>(0));
    hy_host = std::vector<double>(hy_sz, static_cast<double>(0));
    cy_host = std::vector<double>(hy_sz, static_cast<double>(0));

    workspace = std::vector<half>(workSpace_sz, static_cast<half>(0));
    outhost = std::vector<double>(out_sz, static_cast<double>(0));

    srand(0);
    double scale = 0.01;

    for (int i = 0; i < in_sz; i++) {
        in[i] = static_cast<half>((static_cast<double>(scale * rand()) * (1.0 / RAND_MAX)));
    }

    for (int i = 0; i < hy_sz; i++) {
        hx[i] = static_cast<half>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
    }

    for (int i = 0; i < hy_sz; i++) {
        cx[i] = static_cast<half>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
    }

    for (int i = 0; i < wei_sz; i++) {
        wei[i] = static_cast<half>((scale * static_cast<double>((rand()) * (1.0 / RAND_MAX) - 0.5)));
    }

    hipMemcpy(in_dev, in.data(), sizeof(half) * in_sz, hipMemcpyHostToDevice);
    hipMemcpy(wei_dev, wei.data(), sizeof(half) * wei_sz, hipMemcpyHostToDevice);
    hipMemcpy(out_dev, out.data(), sizeof(half) * out_sz, hipMemcpyHostToDevice);
    hipMemcpy(cx_dev, cx.data(), sizeof(half) * hy_sz, hipMemcpyHostToDevice);
    hipMemcpy(hx_dev, hx.data(), sizeof(half) * hy_sz, hipMemcpyHostToDevice);
    hipMemcpy(workspace_dev, workspace.data(), sizeof(half) * workSpace_sz, hipMemcpyHostToDevice);
    hipMemcpy(hy_dev, hy.data(), sizeof(half) * hy_sz, hipMemcpyHostToDevice);
    hipMemcpy(cy_dev, cy.data(), sizeof(half) * hy_sz, hipMemcpyHostToDevice);

    /******************************
    Initialization end
    *****************************/

    miopenRNNForwardInference(handle, rnnDesc, nseq, input_tensors.data(), in_dev, hidden_tensor, hx_dev, hidden_tensor,
                              cx_dev, weight_tensor, wei_dev, output_tensors.data(), out_dev, hidden_tensor, hy_dev,
                              hidden_tensor, cy_dev, workspace_dev, workSpace_sz * sizeof(half));

    hipMemcpy(out.data(), out_dev, sizeof(half) * out_sz, hipMemcpyDeviceToHost);

    half *golden = new half[out_sz];
    std::string test_vector_data = TEST_VECTORS_DATA;

    std::string output_lstm = test_vector_data + "load/gemv/miopen_lstm.dat";

    load_data(output_lstm.c_str(), reinterpret_cast<char *>(golden), sizeof(half) * out_sz);
    ret = compare_data_round_off(golden, out.data(), out_sz);

    miopenDestroyTensorDescriptor(output_tensor);
    miopenDestroyTensorDescriptor(weight_tensor);
    miopenDestroyTensorDescriptor(hidden_tensor);
    miopenDestroyTensorDescriptor(input_tensor);

    miopenDestroyRNNDescriptor(rnnDesc);

    miopenDestroy(handle);

    hipHostFree(&in_dev);
    hipHostFree(&hx_dev);
    hipHostFree(&out_dev);
    hipHostFree(&wei_dev);
    hipHostFree(&cx_dev);
    hipHostFree(&workspace_dev);
    hipHostFree(&hy_dev);
    hipHostFree(&cy_dev);

    free(golden);

    /* __PIM_API__ call : Deinitialize PimRuntime */
    PimDeinitialize();

    return ret;
}

TEST(MIOpenIntegrationTest, MIOpenRnnLstm) { EXPECT_TRUE(miopen_rnn_lstm() == 0); }
