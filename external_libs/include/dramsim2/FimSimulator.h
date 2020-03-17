#ifndef __FIM_SIMULATOR_HPP__
#define __FIM_SIMULATOR_HPP__

#include "FIMCtrl.h"

class FimSimulator {
public:
    FimSimulator();
    void initialize(const string& device_ini_file_name, const string& system_ini_file_name, size_t megs_of_memory,
                    size_t num_fim_chan, size_t num_fim_rank);

    void set_data_for_test(NumpyBurstType* input0, NumpyBurstType* input1, uint16_t* test_input);
    void preload_data(void* data, size_t data_size);
    void read_memory_trace(const string& file_name, vector<MemTraceData>& vec_trace_data);
    void execute_add(void* trace_data, size_t num_trace);
    void alloc_burst(size_t data_size);
    void get_uint16_result(uint16_t* output_data, size_t num_data);
    void run();
    void compare_result(size_t num_data, NumpyBurstType* output_npbst);
    void compare_result_arr(uint16_t* test_output, size_t num_data, NumpyBurstType* output_npbst);
    void vector_to_arr(vector<MemTraceData>& vec_trace_data, MemTraceData* trace_data);
    
private:
    
    void convert_arr_to_burst(void* data, size_t data_size, BurstType* bst);

    void push_trace(vector<TraceDataBst>* trace_bst);
    void convert_to_burst_trace(void* trace_data, vector<TraceDataBst>* trace_bst, size_t num_trace);

private:
    shared_ptr<FIMController> fim_controller_; // for test
    shared_ptr<MultiChannelMemorySystem> mem_;

    BurstType* input_burst_;
    BurstType* output_burst_;
    int bst_size_;
    size_t cycle_;
};


#endif


