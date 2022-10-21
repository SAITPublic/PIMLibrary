#ifndef PIMRANK_H
#define PIMRANK_H

#include <vector>
#include "AddressMapping.h"
#include "BusPacket.h"
#include "Configuration.h"
#include "PIMBlock.h"
#include "PIMCmd.h"
#include "Rank.h"
#include "SimulatorObject.h"

using namespace std;
using namespace DRAMSim;

namespace DRAMSim
{
#define OUTLOG_ALL(msg)                                                                                          \
    msg << " ch" << getChanId() << " ra" << getRankId() << " bg" << config.addrMapping.bankgroupId(packet->bank) \
        << " b" << packet->bank << " r" << packet->row << " c" << packet->column << " @" << currentClockCycle
#define OUTLOG_CH_RA(msg) msg << " ch" << getChanId() << " ra" << getRankId() << " @" << currentClockCycle
#define OUTLOG_PRECHARGE(msg)                                                                                    \
    msg << " ch" << getChanId() << " ra" << getRankId() << " bg" << config.addrMapping.bankgroupId(packet->bank) \
        << " b" << packet->bank << " r" << bankStates[packet->bank].openRowAddress << " @" << currentClockCycle
#define OUTLOG_GRF_A(msg)                                                                                              \
    msg << " ch" << getChanId() << " ra" << getRankId() << " pb" << packet->bank / 2 << " reg" << packet->column - 0x8 \
        << " @" << currentClockCycle
#define OUTLOG_GRF_B(msg)                                                                      \
    msg << " ch" << getChanId() << " ra" << getRankId() << " pb" << packet->bank / 2 << " reg" \
        << packet->column - 0x18 << " @" << currentClockCycle
#define OUTLOG_B_GRF_A(msg) \
    msg << " ch" << getChanId() << " ra" << getRankId() << " reg" << packet->column - 0x8 << " @" << currentClockCycle
#define OUTLOG_B_GRF_B(msg) \
    msg << " ch" << getChanId() << " ra" << getRankId() << " reg" << packet->column - 0x18 << " @" << currentClockCycle
#define OUTLOG_B_CRF(msg) \
    msg << " ch" << getChanId() << " ra" << getRankId() << " idx" << packet->column - 0x4 << " @" << currentClockCycle

class Rank;  // forward declaration
class PIMRank : public SimulatorObject
{
   private:
    int chanId;
    int rankId;
    ostream& dramsimLog;
    Configuration& config;

    int pimPC_;
    int lastJumpIdx_;
    int numJumpToBeTaken_;
    int lastRepeatIdx_;
    int numRepeatToBeDone_;
    bool pimOpMode_;
    bool toggleEvenBank_;
    bool toggleOddBank_;
    bool toggleRa12h_;
    bool useAllGrf_;
    bool crfExit_;

   public:
    PIMRank(ostream& simLog, Configuration& configuration);
    ~PIMRank() {}

    void attachRank(Rank* r);
    int getChanId() const;
    void setChanId(int id);
    int getRankId() const;
    void setRankId(int id);
    void update();

    void readHab(BusPacket* packet);
    void writeHab(BusPacket* packet);
    void doPim(BusPacket* packet);
    void doPimBlock(BusPacket* packet, pimCmd curCmd, int pimblock_id);
    void controlPim(BusPacket* packet);

    void readOpd(int pb, BurstType& bst, pimOpdType type, BusPacket* packet, int idx, bool is_auto, bool is_mac);
    void writeOpd(int pb, BurstType& bst, pimOpdType type, BusPacket* packet, int idx, bool is_auto, bool is_mac);

    vector<pimBlockT> pimBlocks;

    bool isToggleCond(BusPacket* packet);

    union crf_t {
        uint32_t data[32];
        BurstType bst[4];
        crf_t() { memset(data, 0, sizeof(uint32_t) * 32); }
    } crf;

    unsigned inline getGrfIdx(unsigned idx) { return idx & 0x7; }
    unsigned inline getGrfIdxHigh(unsigned r, unsigned c) { return ((r & 0x1) << 2 | ((c >> 3) & 0x3)); }

    Rank* rank;
};
}  // namespace DRAMSim
#endif
