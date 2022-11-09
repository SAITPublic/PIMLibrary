#ifndef _PIM_GEMM_PERF_H_
#define _PIM_GEMM_PERF_H_

#include "common_perf.h"
#include "pim_data_types.h"

class PimGemmTest
{
   public:
    PimGemmTest(unsigned n, unsigned c, unsigned in_h, unsigned in_w, unsigned out_h, unsigned out_w, PimActFunc act,
                bool has_bias, PimGemmOrder gemm_order);
    ~PimGemmTest();
    void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.01f);
    void execute_op(bool block = true);
    void finalize();
    void run_with_explicit_reordering(bool use_device_weight, bool block = true, unsigned niter = 1);
    int validate(float epsilon = 1e-2);
    double get_flt_ops();

   private:
    bool is_support_activation(const PimActFunc& act) { return (act == ACT_RELU || act == NONE) ? true : false; }
    // (n_, c, h, in_w) * (n_, c, in_w, out_w_) = (n_, c, h, out_w_)
    unsigned n_;
    unsigned c_;
    unsigned in_h_;
    unsigned in_w_;
    unsigned out_h_;
    unsigned out_w_;

    PimActFunc act_;
    bool has_bias_;
    PimGemmOrder gemm_order_;

    unsigned in_size_;
    unsigned wgt_size_;
    unsigned out_size_;
    double flt_ops_;

    PimGemmDesc* desc_;
    PimBo *h_i_, *h_w_, *h_b_, *h_o_;  // input, weight, bias, output
    PimBo *d_i_, *d_w_, *d_b_, *d_o_;
    PimBo* golden_;
};

class PimGemmTestFixture : public PerformanceAnalyser
{
   public:
    PimGemmTestFixture();

   protected:
    PimActFunc act;
    bool has_bias;

    int ExecuteTest();
    int ExecuteTestExplicitReordering();
};

#endif
