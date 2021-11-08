/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or
 * computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung
 * Electronics.
 */
#ifndef __PIMC_DRIVER_H__
#define __PIMC_DRIVER_H__
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <pim_compiler.h>
#include <pim_runtime_api.h>
#include <vector>
extern uint64_t g_pim_base_addr;

namespace pim
{
namespace runtime
{
namespace pimc_driver
{
enum class DIMENSIONS { N, C, H, W };

template <class T>
class Tensor
{
   public:
    ~Tensor() = default;
    Tensor(Tensor &&) = delete;
    Tensor(const Tensor &) = delete;
    Tensor &operator=(Tensor &&) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(pimc::TensorDesc desc, T *data) : desc_(desc), data_(data) {}
    T *get_data() { return data_; }
    pimc::TensorDesc get_desc() { return desc_; }
   private:
    pimc::TensorDesc desc_;
    T *data_;
};

class KernelArgs
{
   public:
    ~KernelArgs() { hipFree(crf_binary_device_); };
    KernelArgs() = delete;
    KernelArgs(void *args, size_t size) : size_(size)
    {
        config[1] = args;
        hipMalloc((void **)&crf_binary_device_, 128);
    }

    KernelArgs(void *args, size_t size, std::string crf_binary_host, hipFunction_t kernel)
        : size_(size), crf_binary_host_(crf_binary_host), kernel_(kernel)
    {
        config[1] = args;
        hipMalloc((void **)&crf_binary_device_, 128);
    }
    KernelArgs(KernelArgs &&) = default;
    KernelArgs(const KernelArgs &) = default;
    KernelArgs &operator=(KernelArgs &&) = default;
    KernelArgs &operator=(const KernelArgs &) = default;

    virtual void **get_kconfig() = 0;
    hipFunction_t get_kernel() { return kernel_; }
   protected:
    size_t size_;
    std::string crf_binary_host_;
    hipFunction_t kernel_;

    uint8_t *crf_binary_device_;
    void *config[5] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, nullptr, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size_,
                       HIP_LAUNCH_PARAM_END};
};

template <class T>
class GemvKArgs : public KernelArgs
{
   public:
    ~GemvKArgs() = default;
    GemvKArgs(GemvKArgs &&) = default;
    GemvKArgs(const GemvKArgs &) = default;
    GemvKArgs &operator=(GemvKArgs &&) = default;
    GemvKArgs &operator=(const GemvKArgs &) = default;

    GemvKArgs() : KernelArgs(&args_, sizeof(args_)) {}
    GemvKArgs(Tensor<T> *in, Tensor<T> *out, Tensor<T> *wt, std::string crf_binary, hipFunction_t kernel)
        : KernelArgs(&args_, sizeof(args_), crf_binary, kernel), input_vector_(in), output_vector_(out), weights_(wt)
    {
    }

    void set_input_vector(Tensor<T> *input_vector) { input_vector_ = input_vector; }
    void set_output_vector(Tensor<T> *output_vector) { output_vector_ = output_vector; }
    void set_weights(Tensor<T> *weights) { weights_ = weights; }
    void set_compute_tile(uint32_t compute_tile) { args_.n_compute_tile = compute_tile; }
    void set_memory_tile(uint32_t memory_tile) { args_.n_memory_tile = memory_tile; }
    void set_out_tile(uint32_t out_tile) { args_.n_out_tile = out_tile; }
    void set_gemv_add(uint32_t gemv_add) { args_.is_gemv_add = gemv_add; }
    void set_temp_buffer(uint8_t *buffer) { args_.pim_gemv_tmp_buffer = buffer; };
    Tensor<T> *get_input_vector() { return input_vector_; }
    Tensor<T> *get_output_vector() { return output_vector_; }
    Tensor<T> *get_weights() { return weights_; }
    void **get_kconfig()
    {
        args_.pim_ctr = (uint8_t *)g_pim_base_addr;
        args_.pim_weight = (uint8_t *)weights_->get_data();
        args_.pim_input = (uint8_t *)input_vector_->get_data();
        args_.output = (uint8_t *)output_vector_->get_data();
        args_.batch_dim = input_vector_->get_desc().get_dim(uint32_t(DIMENSIONS::N));
        args_.output_dim = weights_->get_desc().get_dim(uint32_t(DIMENSIONS::H));
        hipMemcpy((void *)crf_binary_device_, (uint8_t *)(crf_binary_host_.c_str()), crf_binary_host_.size(),
                  hipMemcpyHostToDevice);
        args_.crf_binary = crf_binary_device_;
        return reinterpret_cast<void **>(&config);
    }

   private:
    Tensor<T> *input_vector_;
    Tensor<T> *weights_;
    Tensor<T> *output_vector_;

    struct kArgs {
        uint8_t *pim_ctr;
        uint8_t *pim_weight;
        uint8_t *pim_gemv_tmp_buffer;
        uint8_t *pim_input;
        uint8_t *output;
        uint32_t batch_dim;
        uint32_t n_memory_tile;
        uint32_t n_compute_tile;
        uint32_t n_out_tile;
        uint32_t output_dim;
        uint8_t *crf_binary;
        uint32_t is_gemv_add;
    };

    kArgs args_;
};

template <class T>
class EltArgs : public KernelArgs
{
   public:
    ~EltArgs() = default;
    EltArgs(EltArgs &&) = default;
    EltArgs(const EltArgs &) = default;
    EltArgs &operator=(EltArgs &&) = default;
    EltArgs &operator=(const EltArgs &) = default;

    EltArgs() : KernelArgs(&args_, sizeof(args_)) { hipMalloc((void **)&crf_binary_device_, 128); }
    EltArgs(Tensor<T> *in1, Tensor<T> *in2, Tensor<T> *out, std::string crf_binary, hipFunction_t kernel)
        : KernelArgs(&args_, sizeof(args_), crf_binary, kernel),
          input_vector0_(in1),
          input_vector1_(in2),
          output_vector_(out)
    {
    }

    void set_input_vectors(Tensor<T> *input_vector0, Tensor<T> *input_vector1)
    {
        input_vector0_ = input_vector0;
        input_vector1_ = input_vector1;
    }

    void set_output_vector(Tensor<T> *output_vector) { output_vector_ = output_vector; }
    std::vector<Tensor<T> *> get_input_vectors()
    {
        std::vector<Tensor<T> *> ret{input_vector0_, input_vector1_};
        return ret;
    }

    Tensor<T> *get_output_vector() { return output_vector_; }
    void **get_kconfig()
    {
        args_.input0_ = (uint8_t *)input_vector0_->get_data();
        args_.input1_ = (uint8_t *)input_vector1_->get_data();
        args_.g_pim_base_addr_ = (uint8_t *)g_pim_base_addr;
        args_.output_ = (uint8_t *)output_vector_->get_data();
        args_.num_tile_ = (output_vector_->get_desc().get_dim(3)) / (131072);

        hipMemcpy((void *)crf_binary_device_, (uint8_t *)(crf_binary_host_.c_str()), crf_binary_host_.size(),
                  hipMemcpyHostToDevice);
        args_.crf_binary_ = crf_binary_device_;
        return reinterpret_cast<void **>(&config);
    }

   private:
    Tensor<T> *input_vector0_;
    Tensor<T> *input_vector1_;
    Tensor<T> *output_vector_;

    struct kArgs {
        uint8_t *input0_;
        uint8_t *input1_;
        uint8_t *g_pim_base_addr_;
        uint8_t *output_;
        uint32_t num_tile_;
        uint8_t *crf_binary_;
    };

    kArgs args_;
};

template <class T>
class ReluArgs : public KernelArgs
{
   public:
    ~ReluArgs() = default;
    ReluArgs(ReluArgs &&) = default;
    ReluArgs(const ReluArgs &) = default;
    ReluArgs &operator=(ReluArgs &&) = default;
    ReluArgs &operator=(const ReluArgs &) = default;

    ReluArgs() : KernelArgs(&args_, sizeof(args_)) {}
    ReluArgs(Tensor<T> *in, Tensor<T> *out, std::string crf_binary, hipFunction_t kernel)
        : KernelArgs(&args_, sizeof(args_), crf_binary, kernel), input_vector_(in), output_vector_(out)
    {
    }

    void set_input_vectors(Tensor<T> *input_vector) { input_vector_ = input_vector; }
    void set_output_vector(Tensor<T> *output_vector) { output_vector_ = output_vector; }
    void set_num_tile(uint32_t num_tile) { args_.num_tile_ = num_tile; }
    std::vector<Tensor<T> *> get_input_vectors()
    {
        std::vector<Tensor<T> *> ret{input_vector_};
        return ret;
    }

    Tensor<T> *get_output_vector() { return output_vector_; }
    void **get_kconfig()
    {
        args_.input_ = (uint8_t *)input_vector_->get_data();
        args_.g_pim_base_addr_ = (uint8_t *)g_pim_base_addr;
        args_.output_ = (uint8_t *)output_vector_->get_data();

        hipMemcpy((void *)crf_binary_device_, (uint8_t *)(crf_binary_host_.c_str()), crf_binary_host_.size(),
                  hipMemcpyHostToDevice);
        args_.crf_binary_ = crf_binary_device_;
        return reinterpret_cast<void **>(&config);
    }

   private:
    Tensor<T> *input_vector_;
    Tensor<T> *output_vector_;

    struct kArgs {
        uint8_t *input_;
        uint8_t *g_pim_base_addr_;
        uint8_t *output_;
        uint32_t num_tile_;
        uint8_t *crf_binary_;
    };

    kArgs args_;
};
/**
 * @brief Base class for Executors
 */
class Executor
{
   public:
    /**
     * @brief Constructor for Executor class
     *
     * @param pim_op object returned from compiler along with buffers from PIMLibrary
     */
    Executor() = default;
    pimc::PimCCompiled *get_pim_op() { return pim_op_; }
    void set_pim_op(pimc::PimCCompiled *pim_op) { pim_op_ = pim_op; }
    virtual bool execute(PimOpType op_type) = 0;

   private:
    Executor(Executor &&) = delete;
    Executor(const Executor &) = delete;
    Executor &operator=(Executor &&) = delete;
    Executor &operator=(const Executor &) = delete;

    pimc::PimCCompiled *pim_op_;
};

/**
 * @brief Class implementing Executor for PIM
 */
class HIPExecutor : public Executor
{
   public:
    /**
     * @brief Constructor for Pim Executor class
     *
     * @param pim_op pim_op object returned from compiler along with buffers from PIMLibrary
     */
    HIPExecutor() { kargs_ = nullptr; };
    ~HIPExecutor() {}
    void set_kernel_args(KernelArgs *kargs) { kargs_ = kargs; }
    bool execute(PimOpType op_type);

   private:
    HIPExecutor(HIPExecutor &&) = delete;
    HIPExecutor(const HIPExecutor &) = delete;
    HIPExecutor &operator=(HIPExecutor &&) = delete;
    HIPExecutor &operator=(const HIPExecutor &) = delete;

    KernelArgs *kargs_;
};

class HIPCodegen
{
   public:
    HIPCodegen() = default;
    bool execute(PimOpType op, pimc::PimDeviceConfig device_config);
    pimc::PimCCompiled *get_pim_op() { return pim_op_; }
    void set_input_desc(std::vector<pimc::TensorDesc> input_list) { input_list_ = input_list; }
    void set_output_desc(std::vector<pimc::TensorDesc> output_list) { output_list_ = output_list; }
   private:
    HIPCodegen(HIPCodegen &&) = delete;
    HIPCodegen(const HIPCodegen &) = delete;
    HIPCodegen &operator=(HIPCodegen &&) = delete;
    HIPCodegen &operator=(const HIPCodegen &) = delete;

    std::string get_pim_program(PimOpType op);
    pimc::PimCCompiled *pim_op_;
    std::vector<pimc::TensorDesc> input_list_, output_list_;
};

/**
 * @brief Class implementing Executor for PIM
 */
class HIPCompiler
{
   public:
    /**
     * @brief Constructor for HIPCompiler class
     *
     * @param pim_op pim_op object returned from compiler along with buffers from PIMLibrary
     */
    HIPCompiler() {}
    ~HIPCompiler() {}
    void execute(pimc::PimCCompiled *pim_op);
    hipFunction_t get_kernel_function() { return kernel_; }
   private:
    HIPCompiler(HIPCompiler &&) = delete;
    HIPCompiler(const HIPCompiler &) = delete;
    HIPCompiler &operator=(HIPCompiler &&) = delete;
    HIPCompiler &operator=(const HIPCompiler &) = delete;

    hipModule_t module_;
    hipFunction_t kernel_;
};

class PimCDriver
{
   public:
    PimCDriver() = default;
    PimCDriver(PimCDriver &&) = delete;
    PimCDriver(const PimCDriver &) = delete;
    PimCDriver &operator=(PimCDriver &&) = delete;
    PimCDriver &operator=(const PimCDriver &) = delete;

    pimc::PimCCompiled *generate_code(PimOpType op, std::vector<pimc::TensorDesc> input_list,
                                      std::vector<pimc::TensorDesc> output_list);
    hipFunction_t compile_code();
    bool execute_code(KernelArgs *kargs);

    // todo:: Pass HW information from user
    pimc::PimDeviceConfig create_device_config()
    {
        // Define memory configuration and mapping
        pimc::PimMemoryMap hbm2_memorymap(1, 14, 3, 2, 1, 1, 6, 1, 5);
        pimc::PimDeviceInfo vega20_pim_device(8, 8, 8, 4, 128, 16384, 8, 16, 4, 1, 64, 16, 32, 4);

        return pimc::PimDeviceConfig(hbm2_memorymap, vega20_pim_device);
    }

    pimc::TensorDesc create_tensor_desc(std::vector<uint32_t> dims)
    {
        uint32_t num_dims = dims.size();
        return pimc::TensorDesc(num_dims, dims);
    }

   private:
    HIPCodegen codegen_;
    HIPCompiler compile_;
    HIPExecutor executor_;
    PimOpType curr_op_type_;
};
}
}
}
#endif
