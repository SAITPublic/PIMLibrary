#include "utility/fim_util.h"

__host__ void get_fim_block_info(FimBlockInfo* fbi) { memcpy(fbi, &vega20_fbi, sizeof(FimBlockInfo)); }

__host__ __device__ uint32_t mask_by_bit(uint32_t value, uint32_t start, uint32_t end)
{
    int length = start - end + 1;
    value = value >> end;
    return value & ((1 << length) - 1);
}

__host__ __device__ uint64_t addr_gen(uint32_t chan, uint32_t rank, uint32_t bankgroup, uint32_t bank, uint32_t row,
                                      uint32_t col)
{
    /* vega20 memory map info */
    int num_row_bit = 14;
    int num_col_high_bit = 3;
    int num_bank_high_bit = 1;
    int num_bankgroup_bit = 2;
    int num_bank_low_bit = 1;
    int num_chan_bit = 6;
    int num_offset_bit = 5;

    uint64_t addr = 0;

    addr = rank;

    addr <<= num_row_bit;
    addr |= row;

    addr <<= num_col_high_bit;
    addr |= mask_by_bit(col, 4, 2);

    addr <<= num_bank_high_bit;
    addr |= mask_by_bit(bank, 1, 1);

    addr <<= num_bankgroup_bit;
    addr |= bankgroup;

    addr <<= num_bank_low_bit;
    addr |= mask_by_bit(bank, 0, 0);

    addr <<= num_chan_bit - 1;
    addr |= mask_by_bit(chan, num_chan_bit - 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(col, 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(chan, 0, 0);

    addr <<= 1;
    addr |= mask_by_bit(col, 0, 0);

    addr <<= num_offset_bit;

#if TARGET && RADEON7
    /* we assume fim kernel run on vega20(32GB) system */
    /* but SAIT server is vega20(16GB) system */
    /* so upper 2bit should be set as 0 for normal work */
    uint64_t mask = 0x1FFFFFFFF;
    addr &= mask;
#endif

    return addr;
}

__host__ __device__ uint64_t addr_gen_safe(uint32_t chan, uint32_t rank, uint32_t bg, uint32_t bank, uint32_t& row,
                                           uint32_t& col)
{
    FimBlockInfo* fbi = &vega20_fbi;

    while (col >= fbi->num_col / fbi->bl) {
        row++;
        col -= (fbi->num_col / fbi->bl);
    }

    if (row >= fbi->num_row) {
    }

    return addr_gen(chan, rank, bg, bank, row, col);
}

#ifdef EMULATOR
extern uint64_t g_fba;
extern int g_ridx[64];
extern int m_width;
extern FimMemTraceData* g_fmtd16;

__device__ void R_CMD(volatile uint8_t* __restrict__ addr)
{
    int bid = hipBlockIdx_x;
    int tid = hipThreadIdx_x;
    int row = bid * m_width;
    int ridx = row + atomicAdd(&g_ridx[bid], 1);

    g_fmtd16[ridx].block_id = bid;
    g_fmtd16[ridx].thread_id = tid;
    g_fmtd16[ridx].addr = (uint64_t)addr - g_fba;
    g_fmtd16[ridx].cmd = 'R';
}

__device__ void W_CMD(volatile uint8_t* __restrict__ addr)
{
    int bid = hipBlockIdx_x;
    int tid = hipThreadIdx_x;
    int row = bid * m_width;
    int ridx = row + atomicAdd(&g_ridx[bid], 1);

    g_fmtd16[ridx].block_id = bid;
    g_fmtd16[ridx].thread_id = tid;
    g_fmtd16[ridx].addr = (uint64_t)addr - g_fba;
    g_fmtd16[ridx].cmd = 'W';
}

__device__ void W_CMD_R(volatile uint8_t* __restrict__ addr, volatile uint8_t* __restrict__ src)
{
    int bid = hipBlockIdx_x;
    int tid = hipThreadIdx_x;
    int row = bid * m_width;
    int ridx = row + atomicAdd(&g_ridx[bid], 1);

    memcpy(g_fmtd16[ridx].data, (uint8_t*)src, 16);
    g_fmtd16[ridx].block_id = bid;
    g_fmtd16[ridx].thread_id = tid;
    g_fmtd16[ridx].addr = (uint64_t)addr - g_fba;
    g_fmtd16[ridx].cmd = 'W';
}

__device__ void B_CMD(int type)
{
    int row = hipBlockIdx_x * m_width;
    int midx = row + atomicAdd(&g_ridx[hipBlockIdx_x], 1);

    memset(g_fmtd16[midx].data, 0, 16);
    g_fmtd16[midx].block_id = hipBlockIdx_x;
    g_fmtd16[midx].thread_id = hipThreadIdx_x;
    g_fmtd16[midx].addr = 0;
    g_fmtd16[midx].cmd = 'B';

    (type == 0) ? __syncthreads() : __threadfence();
}

#else /* TARGET */

__device__ void R_CMD(volatile uint8_t* __restrict__ addr)
{
    asm volatile("global_load_dwordx4 v[84:87], %0, off, glc, slc" ::"v"(addr) : "v84", "v85", "v86", "v87");
}

__device__ void W_CMD(volatile uint8_t* __restrict__ addr)
{
    asm volatile("global_store_dwordx4 %0, v[80:83], off, glc, slc" ::"v"(addr) : "v80", "v81", "v82", "v83");
}

__device__ void W_CMD_R(volatile uint8_t* __restrict__ addr, volatile uint8_t* __restrict__ src)
{
    ((ulonglong2*)addr)[0] = ((ulonglong2*)src)[0];
}

__device__ void B_CMD(int type)
{
    switch (type) {
        case 0:
            __syncthreads();
            break;
        case 1:
            __threadfence();
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
            break;
        case 2:
            __threadfence();
            break;
        default:
            break;
    }
}
#endif /* EMULATOR */

size_t get_aligned_size(FimDesc* fim_desc, FimMemFlag mem_flag, FimBo* fim_bo)
{
    size_t size;

    int n = fim_desc->bshape.n;
    int c = fim_desc->bshape.c;
    int h = fim_desc->bshape.h;
    int w = fim_desc->bshape.w;

    if (mem_flag == GEMV_INPUT) {
        h = 1;
    } else if (mem_flag == GEMV_WEIGHT) {
        n = 1;
    } else if (mem_flag == GEMV_OUTPUT) {
        w = 1;
    }
    fim_bo->bshape_r = fim_desc->bshape_r;
    fim_bo->bshape = {(uint32_t)w, (uint32_t)h, (uint32_t)c, (uint32_t)n};

    size = n * c * h * w;

    return size;
}

void align_shape(FimDesc* fim_desc, FimOpType op_type)
{
    FimBShape bs = fim_desc->bshape_r;

    if (op_type == OP_GEMV) {
        bs.w = 256 * ceil((float)bs.w / 256);
        bs.h = 4096 * ceil((float)bs.h / 4096);
    } else {
        bs.w = (256 * 1024) * ceil((float)bs.w / (256 * 1024));
    }
    fim_desc->bshape = bs;
}

void pad_data(void* input, FimDesc* fim_desc, FimMemType mem_type, FimMemFlag mem_flag)
{
    if (mem_flag == GEMV_INPUT && mem_type == MEM_TYPE_HOST) {
        if (mem_type == MEM_TYPE_HOST) {
            for (int i = 0; i < fim_desc->bshape.n; i++) {
                for (int j = fim_desc->bshape_r.w; j < fim_desc->bshape.w; j++) {
                    ((half*)input)[i * fim_desc->bshape.w + j] = half(0);
                }
            }
        } else {
            if (fim_desc->bshape_r.w != fim_desc->bshape.w)
                hipMemset(input, 0, fim_desc->bshape.n * fim_desc->bshape.w * sizeof(half));
        }
    }

    if (mem_flag == ELT_OP) {
        int padded_size = fim_desc->bshape.n * fim_desc->bshape.c * fim_desc->bshape.h * fim_desc->bshape.w;
        int real_size = fim_desc->bshape_r.n * fim_desc->bshape_r.c * fim_desc->bshape_r.h * fim_desc->bshape_r.w;
        for (int i = real_size; i < padded_size; i++) {
            ((half*)input)[i] = half(0);
        }
    }
}

void pad_data(void* input, int in_size, int in_nsize, int batch_size, FimMemFlag mem_flag)
{
    if (mem_flag == GEMV_INPUT) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = in_size; j < in_nsize; j++) {
                ((half*)input)[in_nsize * i + j] = half(0);
            }
        }
    } else {
        for (int i = in_size; i < in_nsize; i++) {
            ((half*)input)[i] = half(0);
        }
    }
}
