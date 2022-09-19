#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant PimBlockInfo vega20_pbi = {
    .pim_addr_map = AMDGPU_VEGA20,
    .num_banks = 16,
    .num_bank_groups = 4,
    .num_rank_bit = 1,
    .num_row_bit = 14,
    .num_col_high_bit = 3,
    .num_bank_high_bit = 1,
    .num_bankgroup_bit = 2,
    .num_bank_low_bit = 1,
    .num_chan_bit = 6,
    .num_col_low_bit = 2,
    .num_offset_bit = 5,
    .num_grf = 8,
    .num_grf_A = 8,
    .num_grf_B = 8,
    .num_srf = 4,
    .num_col = 128,
    .num_row = 16384,
    .bl = 4,
    .num_pim_blocks = 8,
    .num_pim_rank = 1,
    .num_pim_chan = 64,
    .trans_size = 32,
    .num_out_per_grf = 16,
};

static __constant uint8_t elt_add_hab_to_hab_pim[32] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

static __constant uint8_t elt_add_hab_pim_to_hab[32] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

uint32_t mask_by_bit_(uint32_t value, uint32_t start, uint32_t end)
{
    int length = start - end + 1;
    value >>= end;
    return value & ((1 << length) - 1);
}

uint64_t addr_gen_(uint32_t chan, uint32_t rank, uint32_t bankgroup, uint32_t bank, uint32_t row, uint32_t col)
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
    addr |= mask_by_bit_(col, 4, 2);

    addr <<= num_bank_high_bit;
    addr |= mask_by_bit_(bank, 1, 1);

    addr <<= num_bankgroup_bit;
    addr |= bankgroup;

    addr <<= num_bank_low_bit;
    addr |= mask_by_bit_(bank, 0, 0);

    addr <<= num_chan_bit - 1;
    addr |= mask_by_bit_(chan, num_chan_bit - 1, 1);

    addr <<= 1;
    addr |= mask_by_bit_(col, 1, 1);

    addr <<= 1;
    addr |= mask_by_bit_(chan, 0, 0);

    addr <<= 1;
    addr |= mask_by_bit_(col, 0, 0);

    addr <<= num_offset_bit;

#if TARGET && RADEON7
    /* we assume pim kernel run on vega20(32GB) system */
    /* but SAIT server is vega20(16GB) system */
    /* so upper 2bit should be set as 0 for normal work */
    uint64_t mask = 0x1FFFFFFFF;
    addr &= mask;
#endif

    return addr;
}

#ifdef EMULATOR
void _R_CMD(__global volatile uint8_t* __restrict__ addr, __global PimMemTracer* __restrict__ emulator_trace)
{
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int row = bid * emulator_trace->m_width;
    int ridx = row + atomic_add(&emulator_trace->g_ridx[bid], 1);

    emulator_trace->g_fmtd16[ridx].block_id = bid;
    emulator_trace->g_fmtd16[ridx].thread_id = tid;
    emulator_trace->g_fmtd16[ridx].addr = (uint64_t)addr - emulator_trace->g_fba;
    emulator_trace->g_fmtd16[ridx].cmd = 'R';
}

void _W_CMD(__global volatile uint8_t* __restrict__ addr, __global PimMemTracer* __restrict__ emulator_trace)
{
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int row = bid * emulator_trace->m_width;
    int ridx = row + atomic_add(&emulator_trace->g_ridx[bid], 1);

    emulator_trace->g_fmtd16[ridx].block_id = bid;
    emulator_trace->g_fmtd16[ridx].thread_id = tid;
    emulator_trace->g_fmtd16[ridx].addr = (uint64_t)addr - emulator_trace->g_fba;
    emulator_trace->g_fmtd16[ridx].cmd = 'W';
}

void _W_CMD_R(__global volatile uint8_t* __restrict__ addr, __global volatile uint8_t* __restrict__ src,
              __global PimMemTracer* __restrict__ emulator_trace)
{
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int row = bid * emulator_trace->m_width;
    int ridx = row + atomic_add(&emulator_trace->g_ridx[bid], 1);

    // memcpy(emulator_trace->g_fmtd16[ridx].data, (uint8_t*)src, 16);
    for (int i = 0; i < 16; i++) {
        emulator_trace->g_fmtd16[ridx].data[i] = src[i];
    }
    emulator_trace->g_fmtd16[ridx].block_id = bid;
    emulator_trace->g_fmtd16[ridx].thread_id = tid;
    emulator_trace->g_fmtd16[ridx].addr = (uint64_t)addr - emulator_trace->g_fba;
    emulator_trace->g_fmtd16[ridx].cmd = 'W';
}

void _W_CMD_R_C(__global volatile uint8_t* __restrict__ addr, __constant volatile uint8_t* __restrict__ src,
                __global PimMemTracer* __restrict__ emulator_trace)
{
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int row = bid * emulator_trace->m_width;
    int ridx = row + atomic_add(&emulator_trace->g_ridx[bid], 1);

    // memcpy(emulator_trace->g_fmtd16[ridx].data, (uint8_t*)src, 16);
    for (int i = 0; i < 16; i++) {
        emulator_trace->g_fmtd16[ridx].data[i] = src[i];
    }
    emulator_trace->g_fmtd16[ridx].block_id = bid;
    emulator_trace->g_fmtd16[ridx].thread_id = tid;
    emulator_trace->g_fmtd16[ridx].addr = (uint64_t)addr - emulator_trace->g_fba;
    emulator_trace->g_fmtd16[ridx].cmd = 'W';
}

void _B_CMD(int type, __global PimMemTracer* __restrict__ emulator_trace)
{
    int row = get_group_id(0) * emulator_trace->m_width;
    int midx = row + atomic_add(&emulator_trace->g_ridx[get_group_id(0)], 1);

    // memset(emulator_trace->g_fmtd16[midx].data, 0, 16);
    for (int i = 0; i < 16; i++) {
        emulator_trace->g_fmtd16[midx].data[i] = 0;
    }
    emulator_trace->g_fmtd16[midx].block_id = get_group_id(0);
    emulator_trace->g_fmtd16[midx].thread_id = get_local_id(0);
    emulator_trace->g_fmtd16[midx].addr = 0;
    emulator_trace->g_fmtd16[midx].cmd = 'B';

    (type == 0) ? barrier(CLK_LOCAL_MEM_FENCE) : mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

#else  /* TARGET */

void _R_CMD(__global volatile uint8_t* __restrict__ addr)
{
    // asm volatile("global_load_dwordx4 v[84:87], %0, off, glc, slc" ::"v"(addr) : "v84", "v85", "v86", "v87");
    ((__global int4*)addr)[0] = 0;
}

void _W_CMD(__global volatile uint8_t* __restrict__ addr)
{
    // asm volatile("global_store_dwordx4 %0, v[80:83], off, glc, slc" ::"v"(addr) : "v80", "v81", "v82", "v83");
    ((__global int4*)addr)[0] = 0;
}

void _W_CMD_R(__global volatile uint8_t* __restrict__ addr, __global volatile uint8_t* __restrict__ src)
{
    ((__global int4*)addr)[0] = ((__global int4*)src)[0];
}

void _W_CMD_R_C(__global volatile uint8_t* __restrict__ addr, __constant volatile uint8_t* __restrict__ src)
{
    ((__global int4*)addr)[0] = ((__constant int4*)src)[0];
}

void _B_CMD(int type)
{
    (type == 0) ? barrier(CLK_LOCAL_MEM_FENCE) : mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}
#endif /* EMULATOR */

#ifdef EMULATOR
#define R_CMD(x) _R_CMD(x, emulator_trace)
#define W_CMD(x) _W_CMD(x, emulator_trace)
#define W_CMD_R(x, y) _W_CMD_R(x, y, emulator_trace)
#define W_CMD_R_C(x, y) _W_CMD_R_C(x, y, emulator_trace)
#define B_CMD(x) _B_CMD(x, emulator_trace)
#else
#define R_CMD(x) _R_CMD(x)
#define W_CMD(x) _W_CMD(x)
#define W_CMD_R(x, y) _W_CMD_R(x, y)
#define B_CMD(x) _B_CMD(x)
#define W_CMD_R_C(x, y) _W_CMD_R_C(x, y)
#endif

#define PARK_IN 1
#define CHANGE_SB_HAB 1
#define PROGRAM_CRF 1
#define CHANGE_HAB_HABPIM 1
#define COMPUTE_ELT_OP 1
#define CHANGE_HABPIM_HAB 1
#define CHANGE_HAB_SB 1
#define PARK_OUT 1

__kernel void elt_op_pim(__global uint8_t* __restrict__ operand0, __global uint8_t* __restrict__ operand1,
                         __global uint8_t* __restrict__ output, __global uint8_t* __restrict__ pim_ctr, int num_tile,
                         __global uint8_t* crf_binary, int crf_size
#ifdef EMULATOR
                         ,
                         __global PimMemTraceData* fmtd16, __global size_t* frd_size, int mt_width,
                         __global PimMemTracer* emulator_trace
#endif
                         )
{
#ifdef EMULATOR
    emulator_trace->g_fba = (uint64_t)pim_ctr;
    emulator_trace->g_fmtd16 = fmtd16;
    emulator_trace->g_ridx[get_group_id(0)] = 0;
    emulator_trace->m_width = mt_width;
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    /*
    gidx : coleasced thread index
    offset : mem address to access the 2nd half of 16 bytes.
    */
    int num_col = 32;
    int num_grf = 8;
    int num_ba = 4;
    int gidx = get_local_id(0) / 2;
    uint64_t offset = (get_local_id(0) % 2) * 0x10;
    uint64_t addr, addr_even, addr_odd;

#if PARK_IN
    addr = addr_gen_(get_group_id(0), 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
#endif

    if (get_local_id(0) < 2) {
#if CHANGE_SB_HAB
        addr = addr_gen_(get_group_id(0), 0, 2, 0, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 2, 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
#if PROGRAM_CRF
        addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x3fff, 0x4 + gidx);
        W_CMD_R(&pim_ctr[addr + offset], crf_binary + offset);
#endif
#if CHANGE_HAB_HABPIM
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R_C(&pim_ctr[addr + offset], elt_add_hab_to_hab_pim + offset);
        R_CMD(&pim_ctr[addr + offset]);
#endif
        B_CMD(1);
    }

    if (get_local_id(0) < 16) {
#if COMPUTE_ELT_OP
        for (int tile_index = 0; tile_index < num_tile; tile_index++) {
            unsigned int loc = tile_index * num_grf + gidx;
            unsigned int row = loc / num_col;
            unsigned int col = loc % num_col;

            addr = addr_gen_(get_group_id(0), 0, 0, 0, row, col);
            addr_even = addr + offset;
            addr_odd = addr_even + 0x2000;

            /*
            compute for even bank
            1. fill to GRFA (opr0).
            2. ADD GRFA and even bank (opr1).
            3. issue NOP which inturn write data to even bank.
            */
            R_CMD(&operand0[addr_even]);
            B_CMD(1);

            R_CMD(&operand1[addr_even]);
            B_CMD(1);

            W_CMD(&output[addr_even]);
            W_CMD(&output[addr_even]);
            R_CMD(&output[addr_even]);
            B_CMD(1);

            /*
            compute for odd bank : same as even bank.
            */
            R_CMD(&operand0[addr_odd]);
            B_CMD(1);

            R_CMD(&operand1[addr_odd]);
            B_CMD(1);

            W_CMD(&output[addr_odd]);
            W_CMD(&output[addr_odd]);

            R_CMD(&output[addr_odd]);

            B_CMD(1);
        }
#endif
    }

    if (get_local_id(0) < 4) {
#if CHANGE_HABPIM_HAB
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R_C(&pim_ctr[addr + offset], elt_add_hab_pim_to_hab + offset);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif

#if CHANGE_HAB_SB
        addr = addr_gen_(get_group_id(0), 0, 0, gidx, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
    }

#if PARK_OUT
    addr = addr_gen_(get_group_id(0), 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
    W_CMD(&pim_ctr[addr + offset]);
#endif

#ifdef EMULATOR
    if (get_group_id(0) == 0 && get_local_id(0) == 0) {
        frd_size[0] = emulator_trace->g_ridx[0];
    }
#endif
}

#define COMPUTE_RELU 1

__kernel void relu_pim_operation(__global uint8_t* __restrict__ pim_data, __global uint8_t* __restrict__ output,
                                 __global uint8_t* __restrict__ pim_ctr, int size, __global uint8_t* crf_binary,
                                 int crf_size
#ifdef EMULATOR
                                 ,
                                 __global PimMemTraceData* fmtd16, __global size_t* frd_size, int mt_width,
                                 __global PimMemTracer* emulator_trace
#endif
                                 )
{
#ifdef EMULATOR
    emulator_trace->g_fba = (uint64_t)pim_ctr;
    emulator_trace->g_fmtd16 = fmtd16;
    emulator_trace->g_ridx[get_group_id(0)] = 0;
    emulator_trace->m_width = mt_width;
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    int trans_size = 32;
    int num_col = 32;
    int num_pim_blocks = 8;
    int num_pim_chan = 64;
    int num_grf = 8;
    int num_ba = 4;
    int out_dim = size / trans_size;
    int num_tile = out_dim / (num_pim_blocks * num_pim_chan * num_grf) / 2;

    int gidx = get_local_id(0) / 2;
    uint64_t offset = (get_local_id(0) % 2) * 0x10;
    uint64_t addr, addr_even, addr_odd;

/* Radeon7(VEGA20) memory is 16GB but our target is 32GB system */
/* so program_crf and chagne_pim_mode functions can not access to over 8GB in our system */

#if PARK_IN
    addr = addr_gen_(get_group_id(0), 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
#endif

    if (get_local_id(0) < 2) {
#if CHANGE_SB_HAB
        addr = addr_gen_(get_group_id(0), 0, 2, 0, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 2, 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
        addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x27ff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
#if PROGRAM_CRF
        addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x3fff, 0x4 + gidx);
        W_CMD_R(&pim_ctr[addr + offset], crf_binary + (get_local_id(0) << 4));
#endif
#if CHANGE_HAB_HABPIM
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R_C(&pim_ctr[addr + offset], elt_add_hab_to_hab_pim + offset);
        R_CMD(&pim_ctr[addr + offset]);
#endif
        B_CMD(1);
    }

#if COMPUTE_RELU
    if (get_local_id(0) < 16) {
        for (int tile_idx = 0; tile_idx < num_tile; tile_idx++) {
            unsigned int loc = tile_idx * num_grf + gidx;
            unsigned int row = loc / num_col;
            unsigned int col = loc % num_col;

            addr = addr_gen_(get_group_id(0), 0, 0, 0, row, col);
            addr_even = addr + offset;
            addr_odd = addr_even + 0x2000;

            R_CMD(&pim_data[addr_even]);
            B_CMD(1);

            W_CMD(&output[addr_even]);
            R_CMD(&output[addr_even]);
            B_CMD(1);

            R_CMD(&pim_data[addr_odd]);
            B_CMD(1);

            W_CMD(&output[addr_odd]);
            R_CMD(&output[addr_odd]);
            B_CMD(1);
        }
    }
#endif

    if (get_local_id(0) < 4) {
#if CHANGE_HABPIM_HAB
        addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
        W_CMD_R_C(&pim_ctr[addr + offset], elt_add_hab_pim_to_hab + offset);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
#if CHANGE_HAB_SB
        addr = addr_gen_(get_group_id(0), 0, 0, gidx, 0x2fff, 0x1f);
        W_CMD(&pim_ctr[addr + offset]);
        R_CMD(&pim_ctr[addr + offset]);
        B_CMD(1);
#endif
    }

#if PARK_OUT
    addr = addr_gen_(get_group_id(0), 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
    W_CMD(&pim_ctr[addr + offset]);
#endif

#ifdef EMULATOR
    if (get_group_id(0) == 0 && get_local_id(0) == 0) {
        frd_size[0] = emulator_trace->g_ridx[0];
    }
#endif
}
