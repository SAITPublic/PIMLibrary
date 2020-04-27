#ifndef _FIM_DATA_TYPE_H_
#define _FIM_DATA_TYPE_H_

#include <stddef.h>
#include "half.hpp"

using half_float::half;

#define EMULATOR

#define __FIM_API__

/** TODO add the following function

inline _Float16 convertF2H(float f_val) { return __float2half(f_val); }

**/

inline float convertH2F(half_float::half h_val) { return half_float::detail::half2float<float>(h_val); }
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

typedef enum __FimMemFlag {
    ELT_OP,
    ELT_FIM_INPUT,
    GEMV_INPUT,
    GEMV_WEIGHT,
    GEMV_OUTPUT
} FimMemFlag;

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
    OP_RELU,
    OP_BN,
    OP_DUMMY,
} FimOpType;

typedef enum __FimPrecision {
    FIM_FP16,
    FIM_INT8,
} FimPrecision;

typedef struct __FimBShape {
    uint32_t w;
    uint32_t h;
    uint32_t c;
    uint32_t n;
} FimBShape;

typedef struct __FimBufferObject {
    FimMemType mem_type;
    FimBShape bshape;
    FimPrecision precision;
    size_t size;
    void* data;
} FimBo;

typedef struct __FimDescriptor {
    FimBShape bshape;
    FimBShape bshape_r;
    FimPrecision precision;
    FimOpType op_type;
} FimDesc;

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
    int num_grf_A;
    int num_grf_B;
    int num_srf;
    int num_col;
    int num_row;
    int bl;
    int num_fim_blocks;
    int num_fim_rank;
    int num_fim_chan;
    int trans_size;
    int num_out_per_grf;
} FimBlockInfo;

typedef struct __FimMemTraceData {
    uint8_t data[32];
    uint64_t addr;
    int block_id;
    int thread_id;
    char cmd;
} FimMemTraceData;

#endif /* _FIM_DATA_TYPE_H_ */
