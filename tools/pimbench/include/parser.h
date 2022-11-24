#ifndef _PIM_PERF_PARSER_H_
#define _PIM_PERF_PARSER_H_

#include <stdio.h>
#include <stdlib.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "utility/pim_log.h"

using namespace boost::program_options;
using namespace std;

class Parser
{
   public:
    Parser();
    int print_help();
    string get_order() { return order; };
    int get_num_iter() { return num_iter; };
    variables_map parse_args(int argc, char* argv[]);
    int get_has_bias() { return has_bias; };
    int get_num_batch() { return num_batch; };
    int get_device_id() { return device_id; };
    string get_platform() { return platform; };
    int get_num_chan() { return num_channels; };
    int get_inp_width() { return input_width; };
    string get_operation() { return operation; };
    int get_out_width() { return output_width; };
    string get_precision() { return precision; };
    int get_inp_height() { return input_height; };
    int get_out_height() { return output_height; };
    string get_act_function() { return act_function; };
    ~Parser(){};

   private:
    bool check_validity();

    options_description desc{"PimBench"};
    string order = "";
    string operation = "";
    string platform = "hip";
    string precision = "fp16";
    string act_function = "relu";
    int num_iter = 2;  // considering the first iter as warm up iter.
    int has_bias = 1;
    int device_id = 0;
    int num_batch = -1;
    int input_width = -1;
    int num_channels = -1;
    int input_height = -1;
    int output_width = -1;
    int output_height = -1;
};

#endif
