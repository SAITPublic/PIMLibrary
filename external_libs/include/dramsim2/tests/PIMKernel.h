#ifndef __PIM_KERNEL_HPP__
#define __PIM_KERNEL_HPP__

#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "SystemConfiguration.h"
#include "tests/KernelAddrGen.h"

using namespace std;
using namespace DRAMSim;

class PIMKernel
{
   public:
    int transaction_size_;
    int num_pim_chans_;
    int num_pim_ranks_;
    int num_grfA_;
    int num_grfB_;
    int num_grf_;
    bool useAllGrf_;

   private:
    unsigned cycle_;
    BurstType null_bst_;
    BurstType bst_hab_pim_;
    BurstType bst_hab_;
    BurstType grfB_reset_;
    BurstType crf_bst_[4];
    BurstType* srf_bst_;
    vector<int> pim_chans_;
    vector<int> pim_ranks_;
    PIMMode mode_;
    shared_ptr<MultiChannelMemorySystem> mem_;
    unsigned num_banks_;
    unsigned num_pim_blocks_;
    unsigned num_bank_groups_;

   public:
    PIMKernel(shared_ptr<MultiChannelMemorySystem> mem, int num_pim_chan, int num_pim_rank)
        : mem_(mem), num_pim_chans_(num_pim_chan), num_pim_ranks_(num_pim_rank), cycle_(0)
    {
        transaction_size_ = getConfigParam(UINT, "BL") * (getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8);  // in byte

        // FIXME: HARDCODED
        num_grfA_ = 8;
        num_grfB_ = 8;
        num_grf_ = num_grfA_;
        useAllGrf_ = false;
        srf_bst_ = NULL;
        bst_hab_pim_.u8Data_[0] = 1;
        bst_hab_pim_.u8Data_[16] = 3;
        bst_hab_pim_.u8Data_[21] = 1;
        bst_hab_.u8Data_[0] = 0;

        grfB_reset_.u8Data_[21] = 1;

        pim_chans_.clear();
        pim_ranks_.clear();
        for (int i = 0; i < num_pim_chans_; i++) pim_chans_.push_back(i);
        for (int i = 0; i < num_pim_ranks_; i++) pim_ranks_.push_back(i);
        mode_ = PIMConfiguration::getPIMMode();
        num_banks_ = getConfigParam(UINT, "NUM_BANKS");
        num_pim_blocks_ = getConfigParam(UINT, "NUM_PIM_BLOCKS");
        num_bank_groups_ = getConfigParam(UINT, "NUM_BANK_GROUPS");

        pim_addr_mgr_ = make_shared<PIMAddrManager>(num_pim_chan, num_pim_rank);
    }

    void setPimCmd(vector<pimCmd>& pim_cmds, KernelType ktype, int num_jump_to_be_taken,
                   int num_jump_to_be_taken_odd_bank, int num_jump_to_be_taken_even_bank);
    void addBarrier();
    void runPim();
    uint64_t getCycle();

    void parkIn();
    void parkOut();
    void changePimMode(dramMode mode1, dramMode mode2);
    void addTransactionAll(bool isWrite, int bg, int bank, int row, int col, const std::string tag, BurstType* bst,
                           bool use_barrier = false, int loopCounter = 1);
    void addTransactionAll(bool isWrite, int bg, int bank, int row, int col, BurstType* bst, bool use_barrier = false,
                           int loopCounter = 1);
    /*
    void preprocessBn(NumpyBurstType* mean_npbst, NumpyBurstType* var_npbst, NumpyBurstType* gamma_npbst,
            NumpyBurstType* beta_npbst, NumpyBurstType* input_npbst, fp16** params, float eps);
    void preprocessSrf(NumpyBurstType* input_npbst, fp16** params, int burst_offset, int num_srf_usage);
    */
    /*
    void programSrf();
    */
    void programCrf(vector<pimCmd>& cmds);
    void setCrf(BurstType* bst, bool pim_op, bool use_all_grf, int crf_toggle_cond, bool grfA_zero, bool grfB_zero);
    unsigned getResultColGemv(int input_dim, int output_dim);
    void changeBank(pimBankType bank_types, int& cidx, int& rank, int& bg, int& bank, unsigned& startingRow,
                    unsigned& startingCol, unsigned& row, unsigned& col);
    void preloadGemv(NumpyBurstType* operand, unsigned starting_row = 0, unsigned starting_col = 0);
    void preloadNoReplacement(NumpyBurstType* operand, unsigned startingRow, unsigned startingCol);
    /*
    void preloadEltwise(NumpyBurstType* operand, pimBankType bank_types, unsigned startingRow,
            unsigned startingCol);
    */
    void executeGemv(NumpyBurstType* w_data, NumpyBurstType* i_data, bool is_tree);
    void executeEltwise(int dim, pimBankType bank_types, KernelType ktype, int input0_row, int result_row,
                        int input1_row = 0);
    void computeGemv(NumpyBurstType* data, int num_input_tiles, int numOutputTile, int inputTile, int outputTile,
                     int batchIdx, pimBankType bank_types);
    void computeAddOrMul(int numTile, int input0Row, int resultRow, int input1Row);
    void computeRelu(int numTile, int input0Row, int resultRow);
    // void computeBn(int numTile, int input0Row, int resultRow);

    void readResult(BurstType* resultBst, pimBankType bank_types, int output_dim, uint64_t baseAddr = 0,
                    unsigned startingRow = 0, unsigned startingCol = 0);

    void readData(BurstType* bst_data, size_t bst_cnt, unsigned starting_row = 0, unsigned starting_col = 0);
    void adderTree(BurstType* result, int output_dim, int numTile, int step, fp16* temp);
    shared_ptr<PIMAddrManager> pim_addr_mgr_;
};

#endif
