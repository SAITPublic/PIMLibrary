#ifndef _FIM_RUNTIME_API_H_
#define _FIM_RUNTIME_API_H_

#include "fim_data_types.h"

__FIM_API__ int FimInitialize(FimRuntimeType rtType = RT_TYPE_HIP, FimPrecision = FIM_FP16);
__FIM_API__ int FimDeinitialize(void);

__FIM_API__ int FimAllocMemory(void** ptr, size_t size, FimMemType memType = MEM_TYPE_FIM);
__FIM_API__ int FimAllocMemory(FimBo* fimBo);

__FIM_API__ int FimFreeMemory(void* ptr, FimMemType memType = MEM_TYPE_FIM);
__FIM_API__ int FimFreeMemory(FimBo* fimBo);

__FIM_API__ int FimConvertDataLayout(void* dst, void* src, size_t size, FimOpType opType);
__FIM_API__ int FimConvertDataLayout(FimBo* dst, FimBo* src, FimOpType opType);
__FIM_API__ int FimConvertDataLayout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType opType);

__FIM_API__ int FimCopyMemory(void* dst, void* src, size_t size, FimMemcpyType cpyType);
__FIM_API__ int FimCopyMemory(FimBo* dst, FimBo* src, FimMemcpyType cpyType);

__FIM_API__ int FimExecute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType);
__FIM_API__ int FimExecute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType opType);
__FIM_API__ int FimExecute(FimBo* output, FimBo* fimData, FimOpType opType);

#endif /* _FIM_RUNTIME_API_H_ */
