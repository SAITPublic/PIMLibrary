#ifndef __FIM_CONTROLLER_HPP__
#define __FIM_CONTROLLER_HPP__

#include <sstream>
#include <vector>
#include "FIMCmd.h"
#include "MultiChannelMemorySystem.h"
#include "SystemConfiguration.h"

using namespace std;
using namespace DRAMSim;

class FIMController
{
   public:
    int transaction_size_;
    int num_transaction_;
    int num_chan_;
    int num_rank_;
    int num_bankgroup_;
    int num_bank_;
    int num_row_;
    int num_col_;
    int num_fim_chan_;
    int num_fim_rank_;
    int num_grf_A_;
    int num_grf_B_;
    int num_grf_;
    bool use_all_grf_;

   private:
    unsigned cycle_;
    int num_chan_bit_;
    int num_rank_bit_;
    int num_col_bit_;
    int num_row_bit_;
    int num_bank_bit_;
    int num_bankgroup_bit_;
    int num_offset_bit_;

    int num_col_low_bit_;
    int num_col_high_bit_;
    int num_bank_low_bit_;
    int num_bank_high_bit_;

    BurstType null_bst;
    BurstType init_bst;
    BurstType bst_hab_fim;
    BurstType bst_hab;
    BurstType grfb_reset;
    BurstType crf_bst[4];
    BurstType* srf_bst;

    vector<int> fim_chans_;
    vector<int> fim_ranks_;
    FIMMode mode_;

    BurstType crf_burst;
    shared_ptr<MultiChannelMemorySystem> mem_;

   public:
    FIMController(shared_ptr<MultiChannelMemorySystem> mem, FIMMode mode, int num_chan, int num_rank, int num_bankgroup,
                  int num_bank, int num_row, int num_col, int num_fim_chan, int num_fim_rank)
        : mem_(mem),
          mode_(mode),
          num_rank_(num_rank),
          num_row_(num_row),
          num_col_(num_col),
          num_chan_(num_chan),
          num_bank_(num_bank),
          num_bankgroup_(num_bankgroup),
          num_fim_chan_(num_fim_chan),
          num_fim_rank_(num_fim_rank),
          num_transaction_(0),
          cycle_(0)
    {
        num_bankgroup_bit_ = dramsim_log2(num_bankgroup);
        num_bank_bit_ = dramsim_log2(num_bank) - dramsim_log2(num_bankgroup);
        num_row_bit_ = dramsim_log2(num_row);
        num_chan_bit_ = dramsim_log2(num_chan);
        num_col_bit_ = dramsim_log2(num_col / BL);
        num_rank_bit_ = dramsim_log2(num_rank);
        num_offset_bit_ = dramsim_log2(BL * JEDEC_DATA_BUS_BITS / 8);
        transaction_size_ = BL * (JEDEC_DATA_BUS_BITS / 8);  // in byte

        num_col_low_bit_ = 2;
        num_col_high_bit_ = num_col_bit_ - num_col_low_bit_;
        num_bank_low_bit_ = num_bank_bit_ / 2;
        num_bank_high_bit_ = num_bank_bit_ - num_bank_low_bit_;

        num_grf_A_ = 8;
        num_grf_B_ = 8;
        num_grf_ = num_grf_A_;
        use_all_grf_ = false;
        srf_bst = NULL;

        bst_hab_fim.u8_data_[0] = 1;
        bst_hab_fim.u8_data_[16] = 3;
        bst_hab_fim.u8_data_[21] = 1;
        bst_hab.u8_data_[0] = 0;

        grfb_reset.u8_data_[21] = 1;
        // print_mmap();

        fim_chans_.clear();
        fim_ranks_.clear();
        for (int i = 0; i < num_fim_chan_; i++) {
            fim_chans_.push_back(i);
        }
        for (int i = 0; i < num_fim_rank_; i++) {
            fim_ranks_.push_back(i);
        }
    }

    unsigned mask_by_bit(unsigned value, int starting_bit, int end_bit);
    void set_fim_cmd(vector<fim_cmd>& fim_cmds, int num_jump_to_be_taken, KernelType ktype);
    void add_barrier();
    void run_fim();
    void print_stats();
    void print_mmap();
    void set_fim_chans(vector<int> fim_chans) { fim_chans_ = fim_chans; }
    void set_fim_ranks(vector<int> fim_ranks) { fim_ranks_ = fim_ranks; }
    uint64_t addr_gen(unsigned chan, unsigned rank, unsigned bankgroup, unsigned bank, unsigned row, unsigned col);
    uint64_t addr_gen_safe(unsigned chan, unsigned rank, unsigned bankgroup, unsigned bank, unsigned& row,
                           unsigned& col);
    void add_transaction_all(bool is_write, int bg, int bank, int row, int col, std::string tag, BurstType* bst,
                             bool use_barrier = false, int loop_counter = 1);
    void add_transaction_all(bool is_write, int bg, int bank, int row, int col, const char* tag, BurstType* bst,
                             bool use_barrier = false, int loop_counter = 1);
    void add_transaction_all(bool is_write, int bg, int bank, int row, int col, BurstType* bst,
                             bool use_barrier = false, int loop_counter = 1);

    void change_fim_mode(fim_mode mode1, fim_mode mode2);
    void change_fim_mode(fim_mode mode1, fim_mode mode2, BurstType& bst);
    void park_in();
    void park_out();
    void program_crf(vector<fim_cmd>& cmds);
    void set_crf(BurstType* bst, bool fim_op, bool use_all_grf, int crf_toggle_cond, bool grf_a_zero, bool grf_b_zero);

    unsigned get_result_col(int dim);
    unsigned get_result_col_gemv(int input_dim, int output_dim);
    int get_num_tile(int dim);
    void change_bank(fim_bank_type bank_type, int& cidx, int& rank, int& bg, int& bank, unsigned& starting_row,
                     unsigned& starting_col, unsigned& row, unsigned& col);
    int set_toggle_condition(fim_bank_type bank_type);
    void preprocess_bn(NumpyBurstType* scale_npbst, NumpyBurstType* shift_npbst, NumpyBurstType* gamma_npbst,
                       NumpyBurstType* beta_npbst, NumpyBurstType* input_npbst, fp16** params);

    void execute_gemv(NumpyBurstType* w_data, NumpyBurstType* i_data);
    void execute_eltwise(int dim, fim_bank_type bank_type, KernelType ktype, int input0_row, int result_row,
                         int input1_row = 0);

    void preload_gemv(NumpyBurstType* operand, unsigned starting_row = 0, unsigned starting_col = 0);
    void preload_eltwise(NumpyBurstType* operand, fim_bank_type bank_type, unsigned starting_row,
                         unsigned starting_col);

    void compute_gemv(NumpyBurstType* data, int num_input_tile, int num_output_tile, int input_tile, int output_tile,
                      int batch_idx, fim_bank_type bank_type);
    void compute_add_or_mul(int num_tile, int input0_row, int result_row, int input1_row);
    void compute_relu(int num_tile, int input0_row, int result_row);
    void compute_bn(int num_tile);

    void read_result(BurstType* result_bst, fim_bank_type bank_type, int output_dim, unsigned starting_row = 0,
                     unsigned starting_col = 0);
    void set_fim_chan(int dim);
    void read_memory_trace(const string& filename);
    void run_trace();
    void convert_to_burst_trace();

    void read_data(BurstType* bst_data, size_t bst_cnt, unsigned starting_row = 0, unsigned starting_col = 0);
    void create_eltwise_vector(NumpyBurstType* operand, uint16_t* data, fim_bank_type bank_type,
                               unsigned starting_row = 0, unsigned starting_col = 0);

    void create_bn_vector(NumpyBurstType* operand, uint16_t* data, unsigned starting_row = 0,
                          unsigned starting_col = 0);
    void create_gemv_vector(NumpyBurstType* operand, uint16_t* data, unsigned starting_row = 0,
                            unsigned starting_col = 0);
    void preload_converted_data(uint16_t* data, int data_size, BurstType* test_burst, int bst_cnt);

    void preload_bn(NumpyBurstType* operand, unsigned starting_row = 0, unsigned starting_col = 0);
    void preprocess_srf(NumpyBurstType* input_npbst, fp16** params, int burst_offset, int num_srf_usage);
    void program_srf();
    void read_result_bn(BurstType* result_bst, int num_ba, int num_ch, int num_w, unsigned starting_row = 0,
                        unsigned starting_col = 0);

    void preload_no_replacement(NumpyBurstType* operand, unsigned starting_row, unsigned starting_col);
};

#endif
