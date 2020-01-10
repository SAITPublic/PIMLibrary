#ifndef FIM_RUNTIME_API_H_
#define FIM_RUNTIME_API_H_

#include "fim_data_types.h"

__FIM_API__
int FimInitialize(FimRuntimeType rtType = RT_TYPE_HIP);

__FIM_API__
int FimDeinitialize(void);

__FIM_API__
int FimAllocMemory(float** ptr, size_t size, FimMemType memType = MEM_TYPE_FIM);

__FIM_API__
int FimFreeMemory(float* ptr, FimMemType memType = MEM_TYPE_FIM);

__FIM_API__
int FimCopyMemory(float* dst, float* src, size_t size, FimMemcpyType cpyType);

__FIM_API__
int FimExecute(float* output, float* operand0, float* operand1, size_t size, FimOpType opType, FimPrecision precision = FIM_FP16);

#endif
