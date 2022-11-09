#ifndef _PIM_ELT_PERF_H_
#define _PIM_ELT_PERF_H_

#include "common_perf.h"
#include "pim_data_types.h"

class PimEltTest
{
   public:
    PimEltTest(unsigned n, unsigned c, unsigned in_h, unsigned in_w, PimPrecision precision);
    ~PimEltTest();
    void prepare(float variation = 0.01f);
    void execute_op(bool block = true);
    void finalize();
    int validate(float epsilon = 1e-5);
    double get_flt_ops();

   private:
    unsigned n_;
    unsigned c_;
    unsigned in_h_;
    unsigned in_w_;
    unsigned out_h_;
    unsigned out_w_;

    unsigned in_size_;
    unsigned out_size_;
    double flt_ops_;

    PimPrecision precision_;
    PimDesc* desc_;
    PimBo *h_i_1_, *h_i_2_, *h_o_;
    PimBo *d_i_1_, *d_i_2_, *d_o_;
    PimBo* golden_;
};

class PimReluTest
{
   public:
    PimReluTest(unsigned n, unsigned c, unsigned i_h, unsigned i_w, PimPrecision precision);
    ~PimReluTest();
    void prepare(float variation = 0.01f);
    void execute_op(bool block = true);
    void finalize();
    int validate(float epsilon = 1e-5);
    void calculate_relu_cpu(half_float::half* input, half_float::half* output, int input_len);
    double get_flt_ops();

   private:
    unsigned n_;
    unsigned c_;
    unsigned in_h_;
    unsigned in_w_;
    unsigned out_h_;
    unsigned out_w_;

    unsigned in_size_;
    unsigned out_size_;
    double flt_ops_;

    PimPrecision precision_;
    PimDesc* desc_;
    PimBo *h_i_, *h_o_;
    PimBo *d_i_, *d_o_;
    PimBo* golden_;
};

class PimEltTestFixture : public PerformanceAnalyser
{
   public:
    PimEltTestFixture(){};

   protected:
    int ExecuteTest();
};

class PimReluTestFixture : public PerformanceAnalyser
{
   public:
    PimReluTestFixture(){};

   protected:
    int ExecuteTest();
};

#endif
