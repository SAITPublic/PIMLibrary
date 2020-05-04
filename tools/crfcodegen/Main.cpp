#include <iostream>
#include "FimCrfBinGen.h"
#include "fim_data_types.h"

FimBlockInfo test_fbi = {
    .fim_addr_map = AMDGPU_VEGA20,
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
    .num_fim_blocks = 8,
    .num_fim_rank = 1,
    .num_fim_chan = 64,
    .trans_size = 32,
    .num_out_per_grf = 16,
};

int main(void)
{
    std::cout << "crf code generator" << std::endl;
    crfgen_offline::FimCrfBinGen fim_gen;
    int crf_size = 0;
    uint8_t buffer[128] = {
        0,
    };

    int input_size = 65536 * sizeof(uint16_t);
    int output_size = 65536 * sizeof(uint16_t);

    fim_gen.gen_binary(OP_ELT_ADD, input_size, output_size, &test_fbi, &buffer[0], &crf_size);
    fim_gen.gen_binary(OP_ELT_MUL, input_size, output_size, &test_fbi, &buffer[0], &crf_size);
    fim_gen.gen_binary(OP_RELU, input_size, output_size, &test_fbi, &buffer[0], &crf_size);
    fim_gen.gen_binary(OP_GEMV, input_size, output_size, &test_fbi, &buffer[0], &crf_size);
    fim_gen.gen_binary(OP_BN, input_size, output_size, &test_fbi, &buffer[0], &crf_size);

    return 0;
}
