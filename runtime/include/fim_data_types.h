#ifndef _FIM_DATA_TYPE_H_
#define _FIM_DATA_TYPE_H_

#include <stddef.h>
#include "hip/hip_fp16.h"

#define EMULATOR

#define __FIM_API__


inline float convertH2F(_Float16 h_val) { return __half2float(h_val); }

inline _Float16 convertF2H(float f_val) { return __float2half(f_val); }

typedef enum __FimRuntimeType {
    RT_TYPE_HIP,
    RT_TYPE_OPENCL,
} FimRuntimeType;

typedef enum __FimAddrMap {
    AMDGPU_VEGA20,
} FimAddrMap;

typedef enum __FimMemType {
    MEM_TYPE_HOST,
    MEM_TYPE_DEVICE,
    MEM_TYPE_FIM,
} FimMemType;

typedef enum __FimMemCpyType {
    HOST_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_FIM,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    DEVICE_TO_FIM,
    FIM_TO_HOST,
    FIM_TO_DEVICE,
    FIM_TO_FIM,
} FimMemCpyType;

typedef enum __FimOpType {
    OP_GEMV,
    OP_ELT_ADD,
    OP_ELT_MUL,
    OP_BATCH_NORM,
    OP_RELU,
    OP_DUMMY,
} FimOpType;

typedef enum __FimPrecision {
    FIM_FP16,
    FIM_INT8,
} FimPrecision;

typedef struct __FimBufferObject {
    size_t size;
    FimMemType mem_type;
    void* data;
} FimBo;

typedef enum __FimBankType {
    EVEN_BANK,
    ODD_BANK,
    ALL_BANK,
} FimBankType;

typedef enum __FimMode {
    SB_MODE,
    HAB_MODE,
    HAB_FIM_MODE,
} FimMode;

typedef struct __FimBlockInfo {
    FimAddrMap fim_addr_map;
    int num_banks;
    int num_bank_groups;
    int num_rank_bit;
    int num_row_bit;
    int num_col_high_bit;
    int num_bank_high_bit;
    int num_bankgroup_bit;
    int num_bank_low_bit;
    int num_chan_bit;
    int num_col_low_bit;
    int num_offset_bit;
    int num_grf;
    int num_col;
    int num_row;
    int bl;
    int num_fim_blocks;
    int num_fim_rank;
    int num_fim_chan;
    int trans_size;
} FimBlockInfo;

typedef struct __FimMemTraceData {
    uint64_t data[2];
    uint64_t addr;
    int block_id;
    int thread_id;
    char cmd;
} FimMemTraceData;

#endif /* _FIM_DATA_TYPE_H_ */
