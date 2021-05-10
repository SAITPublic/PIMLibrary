#include <stdexcept>
#include "pim_runtime_api.h"

#include <torch/script.h>
#include <torch/torch.h>
#include "hip/hip_fp16.h"

#include <miopen/miopen.h>

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
    std::cout << "Launcher for PIM_Lstm" << std::endl;

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

    /*
    PrintHalf("Inputb ",i_data,0);
    PrintHalf("Inputb ",i_data,7);
    PrintHalf("Outputb ", o_data,0);
    PrintHalf("Outputb ",o_data,7);
    PrintHalf("h_data",h_data,0);
    PrintHalf("h_data",h_data,7);
    PrintHalf("c_data",c_data,0);
    PrintHalf("c_data",c_data,7);
    PrintHalf("Weightb " ,w_data,0);
    PrintHalf("Weightb " ,w_data,7);
    */
//    PimSynchronize();
    t.start();

    miopenRNNForwardInference(handle, rnnDesc, nseq, input_tensors.data(), i_data, hidden_tensor, h_data, hidden_tensor,
                              c_data, weight_tensor, w_data, output_tensors.data(), o_data, hidden_tensor, ho_data,
                              hidden_tensor, co_data, ws_data, ws_len);

//    PimSynchronize();
    t.stop();
    std::cout << "RNNfwd Duration: " << t.gettime_ms() << std::endl;

    /*
    PrintHalf("Input ",i_data,0);
    PrintHalf("Input ",i_data,7);
    PrintHalf("Output ", o_data,0);
    PrintHalf("Output ",o_data,7);
    PrintHalf("h_data",h_data,0);
    PrintHalf("h_data",h_data,7);
    PrintHalf("c_data",c_data,0);
    PrintHalf("c_data",c_data,7);
    PrintHalf("Weight " ,w_data,0);
    PrintHalf("Weight " ,w_data,7);
    */

    miopenDestroyTensorDescriptor(output_tensor);
    miopenDestroyTensorDescriptor(weight_tensor);
    miopenDestroyTensorDescriptor(hidden_tensor);
    miopenDestroyTensorDescriptor(input_tensor);

    miopenDestroyRNNDescriptor(rnnDesc);

    miopenDestroy(handle);
}

torch::Tensor py_pim_lstm(torch::Tensor input, torch::Tensor weight, torch::Tensor hidden , torch::Tensor cell , torch::Tensor is_bi_dir , torch::Tensor n_ws_len)
{
        Timer t;

        // Grab the input tensors
        auto input_tensor = torch::flatten(input).data_ptr<at::Half>();
        auto weights_tensor = torch::flatten(weight).data_ptr<at::Half>();
        auto hidden_states_tensor = torch::flatten(hidden).data_ptr<at::Half>();
        auto cell_states_tensor = torch::flatten(cell).data_ptr<at::Half>();


        //Todo , verify if PimCopyMemory is required
        int bi_dir = is_bi_dir.item<int>();;
        //PimCopyMemory((void*)&bi_dir, (void*)bi_dir_tensor.flat<int32>().data(), sizeof(int), DEVICE_TO_HOST);

        int ws_len = n_ws_len.item<int>();;
        //PimCopyMemory((void*)&bi_dir, (void*)bi_dir_tensor.flat<int32>().data(), sizeof(int), DEVICE_TO_HOST);


        // NOTE: We start from 1 as 0th entry is always 2 for 2x memory
        std::vector<int> input_dims;
        for (int i = 1; i < input.dim(); i++) input_dims.push_back(input.size(i));

        std::vector<int> hidden_dims;
        for (int i = 1; i < hidden.dim(); i++) hidden_dims.push_back(hidden.size(i));


        // Todo , figure out correct output siz , why does doc state 2d when there is output for each timestep
        auto hidden_out_tensor = torch::zeros_like(hidden);
        auto ho = hidden_out_tensor.data_ptr<at::Half>();
        auto cell_out_tensor = torch::zeros_like(cell);
        auto co = cell_out_tensor.data_ptr<at::Half>();
        auto ws_out_tensor = torch::zeros({1,ws_len},torch::kF16);
        auto ws = cell_out_tensor.data_ptr<at::Half>();
        auto out_tensor = torch::zeros({2 * input_dims[0], input_dims[1], 2 * hidden_dims[2]},torch::kF16);
        torch::Tensor output_tensor = out_tensor.to(torch::kCUDA);
        auto out_data = output_tensor.data_ptr<at::Half>();

        t.start();
        // PrintHalf("Input received",input.data(),0);
        KernelLauncher(input_tensor, weights_tensor, hidden_states_tensor, cell_states_tensor, input_dims, hidden_dims,
                       bi_dir, ws_len, ws, ho, co, out_data);
        t.stop();
        std::cout << "Kernel Duration: " << t.gettime_ms() << std::endl;

        return output_tensor;
}

static auto registry = torch::RegisterOperators("custom_ops::py_pim_lstm", &py_pim_lstm);

