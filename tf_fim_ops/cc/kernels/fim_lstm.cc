#include <miopen/miopen.h>
#include <iostream>
#include "hip/hip_fp16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

class Timer
{
   public:
    Timer(){};
    void start(const bool enabled = true)
    {
        if (!enabled) return;
        st = std::chrono::steady_clock::now();
    }
    void stop(const bool enabled = true)
    {
        if (!enabled) return;
        et = std::chrono::steady_clock::now();
    }
    float gettime_ms() { return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et - st).count(); }

   private:
    std::chrono::time_point<std::chrono::steady_clock> st;
    std::chrono::time_point<std::chrono::steady_clock> et;
};

void PrintHalf(const char* str, const void* data, int idx)
{
    half x = ((half*)data)[idx];
    std::cout << str << float(x) << std::endl;
}

void KernelLauncher(const void* i_data, const void* w_data, const void* h_data, const void* c_data,
                    std::vector<int> in_len, std::vector<int> hid_len, int bi_dir, int ws_len, void* ws_data,
                    void* ho_data, void* co_data, void* o_data)
{
    std::cout << "Launcher for FIM_Lstm" << std::endl;

    Timer t;
    miopenTensorDescriptor_t input_tensor, hidden_tensor, weight_tensor, output_tensor;
    std::vector<miopenTensorDescriptor_t> input_tensors;
    std::vector<miopenTensorDescriptor_t> output_tensors;
    miopenHandle_t handle;
    miopenRNNDescriptor_t rnnDesc;

    miopenCreate(&handle);

    int nseq = in_len[1];               // Number of iterations to unroll over
    int out_h = 2 * hid_len[2];         // for bidirection
    std::vector<int> out_len({out_h});  // output tensor length

    int batch_size = in_len[0];
    // int batch_size = 1;
    t.start();
    std::array<int, 2> in_lens = {{in_len[0], in_len.back()}};
    std::array<int, 2> out_lens = {{in_len[0], out_len[0]}};
    for (int i = 0; i < nseq; i++) {
        miopenCreateTensorDescriptor(&input_tensor);
        miopenSetTensorDescriptor(input_tensor, miopenHalf, 2, in_lens.data(), nullptr);
        input_tensors.push_back(input_tensor);

        miopenCreateTensorDescriptor(&output_tensor);
        miopenSetTensorDescriptor(output_tensor, miopenHalf, 2, out_lens.data(), nullptr);
        output_tensors.push_back(output_tensor);
    }

    t.stop();

    miopenCreateTensorDescriptor(&hidden_tensor);
    miopenCreateTensorDescriptor(&weight_tensor);
    miopenCreateRNNDescriptor(&rnnDesc);
    miopenSetTensorDescriptor(hidden_tensor, miopenHalf, 3, hid_len.data(), nullptr);

    int layer = hid_len[0] / bi_dir;  // Number of hidden stacks
    int wei_hh = hid_len[2];          // Hidden state length
    miopenRNNMode_t mode = miopenLSTM;
    miopenRNNBiasMode_t biasMode = miopenRNNNoBias;
    miopenRNNDirectionMode_t directionMode;
    directionMode = miopenRNNbidirection;
    miopenRNNInputMode_t inMode = miopenRNNlinear;
    miopenRNNAlgo_t algo = miopenRNNdefault;

    miopenSetRNNDescriptor(rnnDesc, wei_hh, layer, inMode, directionMode, mode, biasMode, algo, miopenHalf);
    miopenGetRNNParamsDescriptor(handle, rnnDesc, input_tensor, weight_tensor, miopenHalf);

    // Todo: can be removed , useful to debug params size.
    if (0) {
        t.start();
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
        std::cout << "In size " << in_sz << std::endl;
        std::cout << "Out size " << out_sz << std::endl;
        std::cout << "Param size " << wei_sz << std::endl;
        std::cout << "Workspace size " << workSpace_sz << std::endl;
        miopenGetRNNParamsSize(handle, rnnDesc, input_tensors[0], &wei_sz, miopenHalf);
        t.stop();
        std::cout << "Duration debug: " << t.gettime_ms() << std::endl;
    }

    // PrintHalf("Inputb ",i_data,0);
    // PrintHalf("Inputb ",i_data,7);
    // PrintHalf("Outputb ", o_data,0);
    // PrintHalf("Outputb ",o_data,7);
    // PrintHalf("h_data",h_data,0);
    // PrintHalf("h_data",h_data,7);
    // PrintHalf("c_data",c_data,0);
    // PrintHalf("c_data",c_data,7);
    // PrintHalf("Weightb " ,w_data,0);
    // PrintHalf("Weightb " ,w_data,7);
    FimSynchronize();
    t.start();
    miopenRNNForwardInference(handle, rnnDesc, nseq, input_tensors.data(), i_data, hidden_tensor, h_data, hidden_tensor,
                              c_data, weight_tensor, w_data, output_tensors.data(), o_data, hidden_tensor, ho_data,
                              hidden_tensor, co_data, ws_data, ws_len);
    FimSynchronize();
    t.stop();
    std::cout << "RNNfwd Duration: " << t.gettime_ms() << std::endl;

    // PrintHalf("Input ",i_data,0);
    // PrintHalf("Input ",i_data,7);
    // PrintHalf("Output ", o_data,0);
    // PrintHalf("Output ",o_data,7);
    // PrintHalf("h_data",h_data,0);
    // PrintHalf("h_data",h_data,7);
    // PrintHalf("c_data",c_data,0);
    // PrintHalf("c_data",c_data,7);
    // PrintHalf("Weight " ,w_data,0);
    // PrintHalf("Weight " ,w_data,7);

    miopenDestroyTensorDescriptor(output_tensor);
    miopenDestroyTensorDescriptor(weight_tensor);
    miopenDestroyTensorDescriptor(hidden_tensor);
    miopenDestroyTensorDescriptor(input_tensor);

    miopenDestroyRNNDescriptor(rnnDesc);

    miopenDestroy(handle);
}

class FimLstmOp : public OpKernel
{
   public:
    explicit FimLstmOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        Timer t;
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<Eigen::half>();

        const Tensor& weights_tensor = context->input(1);
        auto weights = weights_tensor.flat<Eigen::half>();

        const Tensor& hidden_states_tensor = context->input(2);
        auto hidden_states = hidden_states_tensor.flat<Eigen::half>();

        const Tensor& cell_states_tensor = context->input(3);
        auto cell_states = cell_states_tensor.flat<Eigen::half>();

        const Tensor& bi_dir_tensor = context->input(4);
        int bi_dir;
        FimCopyMemory((void*)&bi_dir, (void*)bi_dir_tensor.flat<int32>().data(), sizeof(int), DEVICE_TO_HOST);

        const Tensor& ws_len_tensor = context->input(5);
        int ws_len;
        FimCopyMemory((void*)&ws_len, (void*)ws_len_tensor.flat<int32>().data(), sizeof(int), DEVICE_TO_HOST);

        // NOTE: We start from 1 as 0th entry is always 2 for 2x memory
        std::vector<int> input_dims;
        for (int i = 1; i < input_tensor.dims(); i++) input_dims.push_back(input_tensor.dim_size(i));

        std::vector<int> hidden_dims;
        for (int i = 1; i < hidden_states_tensor.dims(); i++) hidden_dims.push_back(hidden_states_tensor.dim_size(i));

        // Create an output tensor
        Tensor* output_tensor = NULL;
        Tensor* hidden_out_tensor = NULL;
        Tensor* cell_out_tensor = NULL;
        Tensor* ws_out_tensor = NULL;

        // Todo , figure out correct output siz , why does doc state 2d when there is output for each timestep
        TensorShape tshape = TensorShape({2 * input_dims[0], input_dims[1], 2 * hidden_dims[2]});

        OP_REQUIRES_OK(context, context->allocate_output(0, tshape, &output_tensor));
        auto output = output_tensor->flat<Eigen::half>();

        OP_REQUIRES_OK(context, context->allocate_output(1, hidden_states_tensor.shape(), &hidden_out_tensor));
        auto ho = hidden_out_tensor->flat<Eigen::half>();

        OP_REQUIRES_OK(context, context->allocate_output(2, cell_states_tensor.shape(), &cell_out_tensor));
        auto co = cell_out_tensor->flat<Eigen::half>();

        TensorShape wshape = TensorShape({1, ws_len});
        OP_REQUIRES_OK(context, context->allocate_output(3, wshape, &ws_out_tensor));
        auto ws = ws_out_tensor->flat<Eigen::half>();

        t.start();
        // PrintHalf("Input received",input.data(),0);
        KernelLauncher(input.data(), weights.data(), hidden_states.data(), cell_states.data(), input_dims, hidden_dims,
                       bi_dir, ws_len, ws.data(), ho.data(), co.data(), output.data());
        t.stop();
        std::cout << "Kernel Duration: " << t.gettime_ms() << std::endl;
    }
};

REGISTER_KERNEL_BUILDER(Name("FimLstm").Device(DEVICE_GPU), FimLstmOp);
