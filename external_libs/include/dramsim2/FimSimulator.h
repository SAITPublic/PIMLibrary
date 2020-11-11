#ifndef __FIM_SIMULATOR_HPP__
#define __FIM_SIMULATOR_HPP__

#include "FIMCtrl.h"

typedef struct __MemTraceData {
    uint8_t data[32];
    uint64_t addr;
    int block_id;
    int thread_id;
    char cmd;
} MemTraceData;

typedef struct __TraceDataBst {
    BurstType data;
    uint64_t addr;
    int ch;
    char cmd;
} TraceDataBst;

class FimSimulator
{
   public:
    FimSimulator();
    ~FimSimulator();
    void initialize(const string& device_ini_file_name, const string& system_ini_file_name, size_t megs_of_memory,
                    size_t num_fim_chan, size_t num_fim_rank);
    void deinitialize();

    // allocate burst type memory.
    void alloc_burst(size_t preload_size, size_t output_size);
    // Write data to the address in order.
    void preload_data_with_addr(uint64_t addr, void* data, size_t data_size);
    // Execute memory traces. void* must be MemTraceData type.
    void execute_kernel(void* trace_data, size_t num_trace);
    // Read data from address in order. data is stored in output_burst_ variable
    void read_result(uint16_t* output_data, uint64_t addr, size_t data_size);
    // Read data from address. it uses only odd bank.
    void read_result_gemv(uint16_t* output_data, uint64_t addr, size_t data_dim);
    void read_result_bn(uint16_t* output_data, uint64_t addr, int num_batch, int num_ch, int num_w,
                        unsigned starting_row, unsigned starting_col, size_t data_size);
    void eltwise_add(void* output_data, uint16_t* reduced_output, int real_dim, int padded_dim, int num_batch);

    // Todo : This should be integrated with the above functions.
    void execute_kernel_bn(void* trace_data, size_t num_trace, int num_batch, int num_ch, int num_width);
    void preload_data(void* data, size_t data_size);
    void get_uint16_result(uint16_t* output_data, size_t num_data);

   private:
    void run();
    void convert_arr_to_burst(void* data, size_t data_size, BurstType* bst);
    void push_trace(vector<TraceDataBst>* trace_bst);
    void push_trace_bn(vector<TraceDataBst>* trace_bst, int num_batch, int num_ch, int num_width);
    void convert_to_burst_trace(void* trace_data, vector<TraceDataBst>* trace_bst, size_t num_trace);

   private:
    shared_ptr<FIMController> fim_controller_;
    shared_ptr<MultiChannelMemorySystem> mem_;

    BurstType* preload_burst_;
    BurstType* output_burst_;
    int bst_size_;
    size_t cycle_;
};

#endif
