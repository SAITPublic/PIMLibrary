#ifndef __PIM_SIMULATOR_HPP__
#define __PIM_SIMULATOR_HPP__

#include <memory>
#include <string>
#include <vector>
#include "tests/PIMKernel.h"

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

class PimSimulator
{
   public:
    PimSimulator();
    ~PimSimulator();
    void initialize(const string& device_ini_file_name, const string& system_ini_file_name, size_t megs_of_memory,
                    size_t num_pim_chan, size_t num_pim_rank);
    void deinitialize();
    // Write data to the address in order.
    void preload_data_with_addr(uint64_t addr, void* data, size_t data_size);
    // Execute memory traces. void* must be MemTraceData type.
    void execute_kernel(void* trace_data, size_t num_trace);
    // Read data from address in order. data is stored in output_burst_ variable
    void read_result(uint16_t* output_data, uint64_t addr, size_t data_size);
    // Read data from address. it uses only odd bank.
    void read_result_gemv(uint16_t* output_data, uint64_t addr, size_t data_dim);
    void read_result_gemv_tree(uint16_t* output_data, uint64_t addr, size_t output_dim, size_t batch_dim,
                               int num_input_tile);

   private:
    void run();
    void convert_arr_to_burst(void* data, size_t data_size, BurstType* bst);
    void push_trace(vector<TraceDataBst>* trace_bst);
    void convert_to_burst_trace(void* trace_data, vector<TraceDataBst>* trace_bst, size_t num_trace);

   private:
    shared_ptr<PIMKernel> pim_kernel_;
    shared_ptr<MultiChannelMemorySystem> mem_;

    int bst_size_;
    size_t cycle_;
    AddrMapping* addrMapping;
};
#endif
