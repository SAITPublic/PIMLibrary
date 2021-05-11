/*********************************************************************************
 *  Copyright (c) 2010-2011, Elliott Cooper-Balis
 *                             Paul Rosenfeld
 *                             Bruce Jacob
 *                             University of Maryland
 *                             dramninjas [at] gmail [dot] com
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *notice,
 *        this list of conditions and the following disclaimer in the
 *documentation
 *        and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/

#ifndef RANK_H
#define RANK_H

#include "Bank.h"
#include "BankState.h"
#include "BusPacket.h"
#include "PIMBlock.h"
#include "PIMCmd.h"
#include "SimulatorObject.h"
#include "SystemConfiguration.h"

using namespace std;
using namespace DRAMSim;

namespace DRAMSim
{
class MemoryController;  // forward declaration
class Rank : public SimulatorObject
{
   private:
    int chan_id;
    int rank_id;
    ostream& dramsim_log;
    unsigned incomingWriteBank;
    unsigned incomingWriteRow;
    unsigned incomingWriteColumn;
    bool isPowerDown;

   public:
    // functions
    Rank(ostream& dramsim_log_);
    virtual ~Rank();

    void receiveFromBus(BusPacket* packet);
    void check(BusPacket* packet);
    void update_state(BusPacket* packet);
    void execute(BusPacket* packet);

    void check_bank(BusPacketType type, int bank, int row);
    void update_bank(BusPacketType type, int bank, int row, bool target_bank, bool target_bankgroup);

    void attachMemoryController(MemoryController* mc);
    int get_chan_id() const;
    void set_chan_id(int id);
    int get_rank_id() const;
    void set_rank_id(int id);
    void update();
    void powerUp();
    void powerDown();

    void read_sb(BusPacket* packet);
    void read_hab(BusPacket* packet);
    //    void read_pim(BusPacket* packet);
    void write_sb(BusPacket* packet);
    void write_hab(BusPacket* packet);
    //    void write_pim(BusPacket* packet);
    void do_pim(BusPacket* packet);
    void control_pim(BusPacket* packet);

    void read_opd(int fb, BurstType& bst, pim_opd_type type, BusPacket* packet, int idx, bool is_auto, bool is_mac);
    void write_opd(int fb, BurstType& bst, pim_opd_type type, BusPacket* packet, int idx, bool is_auto, bool is_mac);
    void time_checker(BusPacket* packet);
    void start_time_check(BusPacket* packet, string time_tag, uint64_t* timer);
    void end_time_check(BusPacket* packet, string time_tag, uint64_t* timer, uint64_t* total_timer = NULL);

    // fields
    MemoryController* memoryController;
    BusPacket* outgoingDataPacket;
    unsigned dataCyclesLeft;
    bool refreshWaiting;

    // these are vectors so that each element is per-bank
    vector<BusPacket*> readReturnPacket;
    vector<unsigned> readReturnCountdown;

    vector<Bank> banks;
    vector<BankState> bankStates;

    bool pim_timer_on;
    uint64_t pim_start_time;
    uint64_t park_in_time;
    uint64_t park_out_time;
    uint64_t pim_time;
    uint64_t transition_time;
    uint64_t counter_act;
    uint64_t total_transition_time;

    pim_mode mode_;
    vector<pim_block_t> pimblocks;

    int PIM_PC_;
    int last_jump_idx_;
    int num_jump_to_be_taken_;
    int last_repeat_idx_;
    int num_repeat_to_be_done_;

    bool abmr1_e;
    bool abmr1_o;
    bool abmr2_e;
    bool abmr2_o;
    bool sbmr1;
    bool sbmr2;

    bool pim_op_mode;
    bool toggle_even_bank;
    bool toggle_odd_bank;
    bool toggle_ra12h;
    bool use_all_grf;
    bool crf_exit;

    bool is_toggle_cond(BusPacket* packet);

    const char* get_mode_color()
    {
        switch (mode_) {
            case pim_mode::SB:
                return END;
            case pim_mode::HAB:
                return GREEN;
            case pim_mode::HAB_PIM:
                return CYAN;
        }
    }

    union crf_t {
        uint32_t data[32];
        BurstType bst[4];

        crf_t() { memset(this, 0, 32); }
    } crf;
};
}  // namespace DRAMSim
#endif
