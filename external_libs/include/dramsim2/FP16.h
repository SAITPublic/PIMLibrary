#ifndef __HALF__HPP__
#define __HALF__HPP__

#include <cstdint>
#include <cstring>
#include <iostream>
#include "half.h"

using namespace std;

typedef half_float::half fp16;
float convertH2F(fp16 val);
fp16 convertF2H(float val);

union fp16i {
    fp16 fval;
    uint16_t ival;

    fp16i() { ival = 0; }
    fp16i(fp16 x) { fval = x; }
    fp16i(uint16_t x) { ival = x; }
};

bool fp16Equal(fp16 A, fp16 B, int maxUlpsDiff, float maxFsdiff);

#endif
