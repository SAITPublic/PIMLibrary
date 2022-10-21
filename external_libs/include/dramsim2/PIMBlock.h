#ifndef __ALU__HPP__
#define __ALU__HPP__

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include "SystemConfiguration.h"
#include "burst.h"

using namespace std;

namespace DRAMSim
{
class pimBlockT
{
   public:
    pimBlockT() { pimPrecision_ = PIMConfiguration::getPIMPrecision(); }
    pimBlockT(const PIMPrecision& pimPrecision) : pimPrecision_(pimPrecision) {}

    BurstType srf;
    BurstType grfA[8];
    BurstType grfB[8];
    BurstType mOut;
    BurstType aOut;

    void add(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst);
    void mac(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst);
    void mul(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst);
    void mad(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst, BurstType& src2Bst);

    std::string print();

   private:
    PIMPrecision pimPrecision_;
};

}  // namespace DRAMSim
#endif
