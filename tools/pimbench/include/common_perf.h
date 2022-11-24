#ifndef _PIM_COMMON_PERF_H_
#define _PIM_COMMON_PERF_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include "half.hpp"
#include "parser.h"
#include "pim_runtime_api.h"
#include "utility/pim_log.h"

using namespace std;

class PerformanceAnalyser
{
   public:
    int SetUp(Parser* parser);
    void Tick();
    void Tock();
    void SetArgs();
    void TearDown();
    int set_device();
    void print_time_data();
    void print_analytical_data();
    virtual int ExecuteTest() = 0;
    void calculate_gflops(double flt_ops);
    std::chrono::duration<double> calculate_elapsed_time();
    void calculate_avg_time();

   protected:
    int num_iter_;
    int num_batch_;
    double gflops_;
    int device_id_;
    Parser* parser_;
    int input_width_;
    int num_channels_;
    int input_height_;
    int output_width_;
    string operation_;
    int output_height_;
    bool block_ = true;
    PimGemmOrder order_;
    PimPrecision precision_;
    PimRuntimeType platform_;
    std::chrono::duration<double> time_duration_;
    std::chrono::duration<double> start_up_time_;
    std::chrono::duration<double> avg_kernel_time_;
    std::chrono::duration<double> kernel_execution_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_, end_;
};

int compare_data(char* data_A, char* data_b, size_t size);
void set_half_data(half_float::half* buffer, half_float::half value, size_t size);
void set_rand_half_data(half_float::half* buffer, half_float::half variation, size_t size);
int compare_half_Ulps_and_absoulte(half_float::half data_a, half_float::half data_b, int allow_bit_cnt,
                                   float absTolerance = 0.001);
int compare_half_relative(half_float::half* data_a, half_float::half* data_b, int size, float absTolerance = 0.0001);
void addCPU(half_float::half* inp1, half_float::half* inp2, half_float::half* output, int length);
void matmulCPU(half_float::half* input, half_float::half* weight, half_float::half* output, int m, int n, int k,
               half_float::half alpha, half_float::half beta);
void addBiasCPU(half_float::half* output, half_float::half* bias, int size);
void reluCPU(half_float::half* data, int size);

#endif
