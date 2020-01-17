#ifndef FIM_RUNTIME_API_H_
#define FIM_RUNTIME_API_H_

#include "fim_data_types.h"

__FIM_API__
int FimInitialize(FimRuntimeType rtType = RT_TYPE_HIP, FimPrecision = FIM_FP16);

__FIM_API__
int FimDeinitialize(void);

__FIM_API__
int FimAllocMemory(void** ptr, size_t size, FimMemType memType = MEM_TYPE_FIM);

__FIM_API__
int FimFreeMemory(void* ptr, FimMemType memType = MEM_TYPE_FIM);

__FIM_API__
int FimDataReplacement(void* data, size_t size, FimOpType opType);

__FIM_API__
int FimCopyMemory(void* dst, void* src, size_t size, FimMemcpyType cpyType);

__FIM_API__
int FimExecute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType);

#endif
