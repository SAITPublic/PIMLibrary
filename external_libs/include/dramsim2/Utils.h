#ifndef UTILS_H_
#define UTILS_H_

namespace DRAMSim
{
#define Byte2GB(x) ((x) >> 30)
#define Byte2MB(x) ((x) >> 20)
#define Byte2KB(x) ((x) >> 10)

inline unsigned uLog2(unsigned value)
{
    unsigned logBase2 = 0;
    unsigned orig = value;
    value >>= 1;
    while (value > 0) {
        value >>= 1;
        logBase2++;
    }
    if ((unsigned)1 << logBase2 < orig) logBase2++;
    return logBase2;
}

inline bool isPowerOfTwo(unsigned long x) { return (1UL << uLog2(x)) == x; }

};  // namespace DRAMSim

#endif  // UTILS_H
