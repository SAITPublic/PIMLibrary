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

int miopen_rnn_lstm()
{
    void *a_data, *b_data, *c_data, *ref_data;
    miopenTensorDescriptor_t input_tensor, hidden_tensor, weight_tensor, output_tensor;
    std::vector<miopenTensorDescriptor_t> input_tensors;
    std::vector<miopenTensorDescriptor_t> output_tensors;
    miopenHandle_t handle;
    miopenRNNDescriptor_t rnnDesc;
    hipStream_t q;
    hipStream_t s;

    int ret = 0;
    hipStreamCreate(&s);
    miopenCreateWithStream(&handle, s);
    miopenGetStream(handle, &q);

    miopenCreateTensorDescriptor(&input_tensor);
    miopenCreateTensorDescriptor(&hidden_tensor);
    miopenCreateTensorDescriptor(&weight_tensor);
    miopenCreateTensorDescriptor(&output_tensor);
    miopenCreateRNNDescriptor(&rnnDesc);

    workspace_dev = nullptr;
    reservespace_dev = nullptr;

    int nseq = 10;                                          // Number of iterations to unroll over
    int in_h = 1024;                                        // Input Length
    std::vector<int> in_len(4, 4, 4, 3, 3, 3, 2, 2, 2, 1);  // input tensor length
    in_len.push_back(in_h);

    int hid_h = 512;                           // Hidden State Length
    int hid_l = 3 * 2;                         // Number of hidden stacks, *2 for bidirection
    std::vector<int> hid_len({hid_l, hid_h});  // hidden tensor length

    int wei_ih = 1024;  // Input Length
    int wei_hh = 512;   // Hidden state length
    int wei_l = 3;      // Number of hidden stacks
    int wei_bi = 2;     // for bidirection
    int wei_oh = wei_hh * wei_bi;
    int wei_sc = 4;  // for lstm

    std::vector<int> wei_len({wei_bi, wei_l, wei_ih, wei_hh, wei_oh, wei_sc});  // weight tensor length

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

    int layer = 3;  // Number of hidden stacks
    miopenRNNMode_t mode = miopenLSTM;
    miopenRNNBiasMode_t biasMode = miopenRNNNoBias;
    miopenRNNDirectionMode_t directionMode;
    directionMode = miopenRNNbidirection;
    miopenRNNInputMode_t inMode = miopenRNNlinear;
    miopenRNNAlgo_t algo = miopenRNNdefault;

    miopenSetRNNDescriptor(rnnDesc, wei_hh, layer, inMode, directionMode, mode, biasMode, algo, data_type);
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

    uint32_t ctx = 0;
    hipMalloc(&in_dev, sizeof(half) * in_sz);
    hipMalloc(&hx_dev, sizeof(half) * hy_sz);
    hipMalloc(&out_dev, sizeof(half) * out_sz);
    hipMalloc(&wei_dev, sizeof(half) * wei_sz);
    hipMalloc(&cx_dev, sizeof(half) * hy_sz);
    hipMalloc(&workspace_dev, sizeof(half) * workSpace_sz);

    hipMalloc(&hy_dev, sizeof(half) * hy_sz);
    hipMalloc(&cy_dev, sizeof(half) * hy_sz);

    /*if(inflags.GetValueInt("forw") != 1)
    {
        din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        dwei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
        dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        dhx_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
        dcx_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
        dhy_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
        dcy_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
    }*/

    in = std::vector<half>(in_sz);
    hx = std::vector<half>(hy_sz, static_cast<Tgpu>(0));
    wei = std::vector<half>(wei_sz);
    out = std::vector<half>(out_sz, static_cast<Tgpu>(0));
    cx = std::vector<half>(hy_sz, static_cast<Tgpu>(0));

    hy = std::vector<half>(hy_sz, static_cast<half>(0));
    cy = std::vector<half>(hy_sz, static_cast<half>(0));
    hy_host = std::vector<half>(hy_sz, static_cast<half>(0));
    cy_host = std::vector<half>(hy_sz, static_cast<half>(0));

    workspace = std::vector<half>(workSpace_sz, static_cast<half>(0));
    outhost = std::vector<double>(out_sz, static_cast<double>(0));
    workspace_host = std::vector<double>(workSpace_sz, static_cast<double>(0));

    std::size_t inputBatchLenSum = std::accumulate(in_len.begin(), in_len.begin() + nseq, 0);

    int hid_h = inflags.GetValueInt("hid_h");
    int layer = inflags.GetValueInt("num_layer");
    int bidir = inflags.GetValueInt("bidirection");

    /*if(inflags.GetValueInt("forw") != 1)
    {
        din       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        dwei      = std::vector<Tgpu>(wei_sz, static_cast<Tgpu>(0));
        dout      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        dhx       = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        dcx       = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        dhy       = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        dcy       = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        din_host  = std::vector<Tref>(in_sz, static_cast<Tref>(0));
        dwei_host = std::vector<Tref>(wei_sz, static_cast<Tref>(0));
        dhx_host  = std::vector<Tref>(hy_sz, static_cast<Tref>(0));
       dcx_host  = std::vector<Tref>(hy_sz, static_cast<Tref>(0));
    }*/

    double scale = 0.01;

    for (int i = 0; i < in_sz; i++) {
        in[i] = 1;
    }

    for (int i = 0; i < hy_sz; i++) {
        hx[i] = 1;
    }

    for (int i = 0; i < hy_sz; i++) {
        cx[i] = 1;
    }

    /*if(inflags.GetValueInt("forw") != 1)
    {
        for(int i = 0; i < out_sz; i++)
        {
            dout[i] = static_cast<Tgpu>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
        }

        for(int i = 0; i < hy_sz; i++)
        {
            dhy[i] = static_cast<Tgpu>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
        }

        if((inflags.GetValueStr("mode")) == "lstm")
        {
            for(int i = 0; i < hy_sz; i++)
            {
                dcy[i] =
                    static_cast<Tgpu>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
            }
        }
    }*/

    for (int i = 0; i < wei_sz; i++) {
        wei[i] = 1;
    }

    hipMemcpy(in_dev, in.data());
    hipMemcpy(wei_dev, wei.data());
    hipMemcpy(out_dev, out.data());
    hipMemcpy(cx_dev, cx.data());
    hipMemcpy(hx_dev, hx.data());
    hipMemcpy(workspace_dev, workspace.data());

    hipMemcpy(hy_dev, hy.data());
    hipMemcpy(cy_dev, cy.data());

    miopenRNNForwardInference(GetHandle(), rnnDesc, adjustedSeqLen, inputTensors.data(), in_dev->GetMem(), hiddenTensor,
                              hx_dev->GetMem(), hiddenTensor, cx_dev->GetMem(), weightTensor, wei_dev->GetMem(),
                              outputTensors.data(), out_dev->GetMem(), hiddenTensor, hy_dev->GetMem(), hiddenTensor,
                              cy_dev->GetMem(), workspace_dev->GetMem(), workspace_dev->GetSize());

    miopenDestroy(handle);

    return ret;
}

TEST(MIOpenIntegrationTest, MIOpenRnnLstm) { EXPECT_TRUE(miopen_rnn_lstm() == 0); }
