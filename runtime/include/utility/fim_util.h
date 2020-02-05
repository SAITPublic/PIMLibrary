#ifndef _FIM_UTIL_H_
#define _FIM_UTIL_H_

#include "fim_data_types.h"

int initTestVector(FimBo& hostInput, FimBo& hostWeight, FimBo& hostOutput);
int verifyResult(FimBo& hostInput, FimBo& hostWeight, FimBo& hostOutput, FimBo& deviceOutput);
bool compareData(float& a, FP16& b, float allow_diff);

#endif /* _FIM_UTIL_H_ */
