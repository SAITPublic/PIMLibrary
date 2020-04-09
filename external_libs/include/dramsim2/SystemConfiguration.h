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

#ifndef SYSCONFIG_H
#define SYSCONFIG_H

#include "PrintMacros.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#endif


// number of latencies per bucket in the latency histogram
// TODO: move to system ini file
#define HISTOGRAM_BIN_SIZE 10

extern std::ofstream cmd_verify_out; // used by BusPacket.cpp if VERIFICATION_OUTPUT is enabled
// extern std::ofstream visDataOut;

// TODO: namespace these to DRAMSim::
extern bool VERIFICATION_OUTPUT; // output suitable to feed to modelsim

extern bool DEBUG_TRANS_Q;
extern bool DEBUG_CMD_Q;
extern bool DEBUG_ADDR_MAP;
extern bool DEBUG_BANKSTATE;
extern bool DEBUG_BUS;
extern bool DEBUG_BANKS;
extern bool DEBUG_POWER;
extern bool USE_LOW_POWER;
extern bool VIS_FILE_OUTPUT;
extern bool PRINT_CHAN_STAT;
extern bool DEBUG_FIM_TIME;
extern bool DEBUG_CMD_TRACE;
extern bool DEBUG_FIM_BLOCK;

extern uint64_t TOTAL_STORAGE;
extern unsigned NUM_BANKS;
extern unsigned NUM_BANKS_LOG;
extern unsigned NUM_RANKS;
extern unsigned NUM_RANKS_LOG;
extern unsigned NUM_CHANS;
extern unsigned NUM_CHANS_LOG;
extern unsigned NUM_ROWS;
extern unsigned NUM_ROWS_LOG;
extern unsigned NUM_COLS;
extern unsigned NUM_COLS_LOG;
extern unsigned NUM_FIM_BLOCKS;
extern unsigned FENCE_CYCLE;

/* ---------------------- */

extern unsigned NUM_BANK_GROUPS;
extern unsigned NUM_BANK_GROUPS_LOG;

/* ---------------------- */

extern unsigned DEVICE_WIDTH;
extern unsigned BYTE_OFFSET_WIDTH;
extern unsigned TRANSACTION_SIZE;
extern unsigned THROW_AWAY_BITS;
extern unsigned COL_LOW_BIT_WIDTH;

extern unsigned tRFC;
extern unsigned tRFCSB;
extern unsigned tREFI;
extern unsigned tREFISB;

extern float tCK;

extern unsigned CL;
extern unsigned AL;

extern unsigned RL;
extern unsigned WL;

extern unsigned BL;
extern unsigned tRAS;
extern unsigned tRCD;

extern unsigned tRCDRD;
extern unsigned tRCDWR;

extern unsigned tRRD;
extern unsigned tRC;
extern unsigned tRP;
extern unsigned tCCD;
extern unsigned tRTP;
extern unsigned tWTR;
extern unsigned tWR;
extern unsigned tRTRS;
extern unsigned XAW;
extern unsigned tXAW;
extern unsigned tCKE;
extern unsigned tXP;

extern unsigned tCMD;

extern unsigned tCCDL;
extern unsigned tCCDS;
extern unsigned tRRDL;
extern unsigned tRRDS;
extern unsigned tWTRL;
extern unsigned tWTRS;
extern unsigned tRTPL;
extern unsigned tRTPS;

extern std::string SIM_TRACE_FILE;
extern std::string MEM_TRACE_FILE;
extern bool SHOW_SIM_OUTPUT;
extern bool LOG_OUTPUT;
extern bool MEM_TRACE_OUTPUT;

#define READ_TO_PRE_DELAY (AL + BL / 2 + max(tRTPL, tCCDL) - tCCDL)
#define READ_TO_PRE_DELAY_LONG (AL + BL / 2 + max(tRTPL, tCCDL) - tCCDL)
#define READ_TO_PRE_DELAY_SHORT (AL + BL / 2 + max(tRTPS, tCCDS) - tCCDS)
#define WRITE_TO_PRE_DELAY (WL + BL / 2 + tWR)
#define READ_TO_WRITE_DELAY (RL + BL / 2 + tRTRS - WL)
#define READ_AUTOPRE_DELAY (AL + tRTP + tRP)
#define WRITE_AUTOPRE_DELAY (WL + BL / 2 + tWR + tRP)

#define WRITE_TO_READ_DELAY_B_LONG (WL + BL / 2 + tWTRL)  // interbank
#define WRITE_TO_READ_DELAY_B_SHORT (WL + BL / 2 + tWTRS) // interbank

#define WRITE_TO_READ_DELAY_R max((int)(WL + BL / 2 + tRTRS) - (int)RL, (int)0) // interrank

/*
#define READ_TO_PRE_DELAY_LONG (AL + BL / 2 + max(tRTPL, tCCDL) - tCCDL)
#define READ_TO_PRE_DELAY_SHORT (AL + BL / 2 + max(tRTPS, tCCDS) - tCCDS)
#define READ_TO_PRE_DELAY (AL + BL / 2 + max(tRTPL, tCCDL) - tCCDL)
#define READ_TO_WRITE_DELAY (RL + BL / 2 + tRTRS - WL)


#define WRITE_TO_READ_DELAY_B  (WL + BL / 2 + tWTRL)
#define WRITE_TO_READ_DELAY_B_LONG (WL + BL / 2 + tWTRL)  // interbank
#define WRITE_TO_READ_DELAY_B_SHORT (WL + BL / 2 + tWTRS) // interbank
#define WRITE_TO_READ_DELAY_R (max(((int)WL + (int)BL / 2 + (int)tRTRS - (int)RL), (int)0))
*/
/* For power parameters (current and voltage), see externs in
 * MemoryController.cpp */

extern unsigned NUM_DEVICES;
/*
#define WRITE_TO_PRE_DELAY (WL + BL / 2 + tWR)
#define READ_AUTOPRE_DELAY (AL + tRTP + tRP)
#define WRITE_AUTOPRE_DELAY (WL + BL / 2 + tWR + tRP)
*/
extern unsigned JEDEC_DATA_BUS_BITS;

// Memory Controller related parameters
extern unsigned TRANS_QUEUE_DEPTH;
extern unsigned CMD_QUEUE_DEPTH;

extern unsigned EPOCH_LENGTH;

extern unsigned TOTAL_ROW_ACCESSES;

extern std::string FIM_PRECISION;

extern std::string ROW_BUFFER_POLICY;
extern std::string SCHEDULING_POLICY;
extern std::string ADDRESS_MAPPING_SCHEME;
extern std::string QUEUING_STRUCTURE;

extern unsigned IDD0;
extern unsigned IDD1;
extern unsigned IDD2P;
extern unsigned IDD2Q;
extern unsigned IDD2N;
extern unsigned IDD3Pf;
extern unsigned IDD3Ps;
extern unsigned IDD3N;
extern unsigned IDD4W;
extern unsigned IDD4R;
extern unsigned IDD5;
extern unsigned IDD6;
extern unsigned IDD6L;
extern unsigned IDD7;

extern unsigned IDD0C;
extern unsigned IDD0Q;

extern unsigned IDD3NC;
extern unsigned IDD3NQ;

extern unsigned IDD4WC;
extern unsigned IDD4WQ;

extern unsigned IDD4RC;
extern unsigned IDD4RQ;

extern float Vdd;

extern float Vddc;
extern float Vddq;
extern float Vpp;

extern unsigned Ealu;
extern unsigned Ereg;

namespace DRAMSim {
    
enum TraceType { k6, mase, misc };

enum AddressMappingScheme {
    Scheme1,
    Scheme2,
    Scheme3,
    Scheme4,
    Scheme5,
    Scheme6,
    Scheme7,
    EagleScheme,
    GPUScheme,
    VegaScheme
};



// used in MemoryController and CommandQueue
enum RowBufferPolicy { OpenPage, ClosePage };

// Only used in CommandQueue
enum QueuingStructure { PerRank, PerRankPerBank };
enum SchedulingPolicy { RankThenBankRoundRobin, BankThenRankRoundRobin };

enum FIMMode { mac_in_bankgroup, mac_in_bank };
enum FIMPrecision { FP16, INT8, FP32 };

enum class fim_mode { SB, HAB, HAB_FIM };

enum class fim_bank_type { EVEN_BANK, ODD_BANK, ALL_BANK };
enum TestKernelType{ ADD_16, ADD_16_NOP, ADD_8, ADD_8_NOP, BN_16, BN_8, ADD_ONE_BANK_8, ADD_ONE_BANK_8_NOP, ADD_ONE_BANK_16, RELU_8 };
enum KernelType{ ADD, BN, RELU, GEMV, RESNET50, MUL, RELU_2BANK};
// set by IniReader.cpp

    
typedef void (*returnCallBack_t)(unsigned id, uint64_t addr, uint64_t clockcycle);
typedef void (*powerCallBack_t)(double bgpower, double burstpower, double refreshpower, double actprepower);

extern RowBufferPolicy rowBufferPolicy;
extern SchedulingPolicy schedulingPolicy;
extern AddressMappingScheme addressMappingScheme;
extern QueuingStructure queuingStructure;
extern FIMMode fimMode;
extern FIMPrecision fimPrecision;
extern int fimDataLength;

//
// FUNCTIONS
//

unsigned inline dramsim_log2(unsigned value) {
    unsigned logbase2 = 0;
    unsigned orig = value;
    value >>= 1;
    while (value > 0) {
        value >>= 1;
        logbase2++;
    }
    if ((unsigned)1 << logbase2 < orig)
        logbase2++;
    return logbase2;
}
inline bool isPowerOfTwo(unsigned long x) { return (1UL << dramsim_log2(x)) == x; }
}; // namespace DRAMSim

#endif
