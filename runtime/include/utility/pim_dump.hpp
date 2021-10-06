/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_DUMP_HPP_
#define _PIM_DUMP_HPP_

#include "half.hpp"
#include "manager/PimInfo.h"
#include "pim_data_types.h"
#include "stdio.h"

inline const char* get_pim_op_string(PimOpType op_type)
{
    const char* op_str;

    switch (op_type) {
        case OP_GEMV:
            op_str = "gemv";
            break;
        case OP_ELT_ADD:
            op_str = "elt_add";
            break;
        case OP_ELT_MUL:
            op_str = "elt_mul";
            break;
        case OP_BN:
            op_str = "bn";
            break;
        case OP_RELU:
            op_str = "relu";
            break;
        case OP_DUMMY:
            op_str = "dummy";
            break;
        default:
            op_str = "None";
            break;
    }

    return op_str;
}

inline int load_data(const char* filename, char* data, size_t size)
{
    FILE* fp = fopen(filename, "r");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        fscanf(fp, "%c", &data[i]);
    }
    fclose(fp);

    return 0;
}

inline int dump_data(const char* filename, char* data, size_t size)
{
    FILE* fp = fopen(filename, "wb");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        fprintf(fp, "%c", data[i]);
    }
    fclose(fp);

    return 0;
}

inline int dump_hexa_array(const char* filename, char* data, size_t size)
{
    FILE* fp = fopen(filename, "wb");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        fprintf(fp, "0x%X, ", (unsigned char)data[i]);
    }
    fclose(fp);

    return 0;
}

template <int block_size>
inline int dump_fmtd(const char* filename, PimMemTraceData* fmtd, size_t fmtd_size)
{
    FILE* fp = fopen(filename, "wb");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < fmtd_size; i++) {
        fprintf(fp, "%d\t%d\t%d\t%c\t0x%lX", i, fmtd[i].block_id, fmtd[i].thread_id, fmtd[i].cmd, fmtd[i].addr);
        if (fmtd[i].cmd == 'W') {
            fprintf(fp, "\t");
            for (int d = 0; d < block_size; d++) fprintf(fp, "%02X", fmtd[i].data[d]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}

inline int compare_data(char* data_a, char* data_b, size_t size)
{
    int ret = 0;

    for (int i = 0; i < size; i++) {
        // std::cout << (int)data_a[i] << " : " << (int)data_b[i] << std::endl;
        if (data_a[i] != data_b[i]) {
            ret = -1;
            break;
        }
    }
    return ret;
}

inline int compare_half_Ulps_and_absoulte(half_float::half data_a, half_float::half data_b, int allow_bit_cnt,
                                          float absTolerance = 0.001)
{
    uint16_t Ai = *((uint16_t*)&data_a);
    uint16_t Bi = *((uint16_t*)&data_b);

    float diff = fabs((float)data_a - (float)data_b);
    int maxUlpsDiff = 1 << allow_bit_cnt;

    if (diff <= absTolerance) {
        return true;
    }

    if ((Ai & (1 << 15)) != (Bi & (1 << 15))) {
        if (Ai == Bi) return true;
        return false;
    }

    // Find the difference in ULPs.
    int ulpsDiff = abs(Ai - Bi);

    if (ulpsDiff <= maxUlpsDiff) return true;

    return false;
}

inline int compare_half_relative(half_float::half* data_a, half_float::half* data_b, int size,
                                 float absTolerance = 0.0001)
{
    int pass_cnt = 0;
    int warning_cnt = 0;
    int fail_cnt = 0;
    int ret = 0;
    std::vector<int> fail_idx;
    std::vector<float> fail_data_pim;
    std::vector<float> fail_data_goldeny;

    float max_diff = 0.0;
    int pass_bit_cnt = 4;
    int warn_bit_cnt = 8;

    for (int i = 0; i < size; i++) {
        if (compare_half_Ulps_and_absoulte(data_a[i], data_b[i], pass_bit_cnt)) {
            // std::cout << "c data_a : " << (float)data_a[i] << " data_b : "  <<(float)data_b[i]  << std::endl;
            pass_cnt++;
        } else if (compare_half_Ulps_and_absoulte(data_a[i], data_b[i], warn_bit_cnt, absTolerance)) {
            // std::cout << "w data_a : " << (float)data_a[i] << " data_b : "  <<(float)data_b[i]  << std::endl;
            warning_cnt++;
        } else {
            if (abs(float(data_a[i]) - float(data_b[i])) > max_diff) {
                max_diff = abs(float(data_a[i]) - float(data_b[i]));
            }
            //  std::cout << "f data_a : " << (float)data_a[i] << " data_b : "  <<(float)data_b[i]  << std::endl;

            fail_idx.push_back(pass_cnt + warning_cnt + fail_cnt);
            fail_data_pim.push_back((float)data_a[i]);
            fail_data_goldeny.push_back((float)data_b[i]);

            fail_cnt++;
            ret = 1;
        }
    }

    int quasi_cnt = pass_cnt + warning_cnt;
    if (ret) {
        printf("relative - pass_cnt : %d, warning_cnt : %d, fail_cnt : %d, pass ratio : %f, max diff : %f\n", pass_cnt,
               warning_cnt, fail_cnt,
               ((float)quasi_cnt / ((float)fail_cnt + (float)warning_cnt + (float)pass_cnt) * 100), max_diff);
#ifdef DEBUG_PIM
        for (int i = 0; i < fail_idx.size(); i++) {
            std::cout << fail_idx[i] << " pim : " << fail_data_pim[i] << " golden :" << fail_data_goldeny[i]
                      << std::endl;
        }
#endif
    }

    return ret;
}

inline void addressMapping(uint64_t physicalAddress, unsigned& newTransactionChan, unsigned& newTransactionRank,
                           unsigned& newTransactionBank, unsigned& newTransactionRow, unsigned& newTransactionColumn)
{
    uint64_t tempA, tempB;
    uint64_t channelBitWidth = 6;
    uint64_t rankBitWidth = 1;
    uint64_t bankBitWidth = 2;
    uint64_t bankgroupBitWidth = 2;
    uint64_t rowBitWidth = 14;
    uint64_t colBitWidth = 5;

    uint64_t byteOffsetWidth = 5;
    physicalAddress >>= byteOffsetWidth;

    uint64_t colLowBitWidth = 2;

    physicalAddress >>= colLowBitWidth;

    uint64_t colHighBitWidth = colBitWidth - colLowBitWidth;

    int col_low_width = 2;
    int ba_low_width = (bankBitWidth - bankgroupBitWidth) / 2;
    int col_high_width = colHighBitWidth - col_low_width;
    int ba_high_width = (bankBitWidth - bankgroupBitWidth) - ba_low_width;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> 1;
    tempB = physicalAddress << 1;
    newTransactionColumn = tempA ^ tempB;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> 1;
    tempB = physicalAddress << 1;
    newTransactionChan = tempA ^ tempB;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> (col_low_width - 1);
    tempB = physicalAddress << (col_low_width - 1);
    newTransactionColumn |= (tempA ^ tempB) << 1;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> (channelBitWidth - 1);
    tempB = physicalAddress << (channelBitWidth - 1);
    newTransactionChan |= (tempA ^ tempB) << 1;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> ba_low_width;
    tempB = physicalAddress << ba_low_width;
    newTransactionBank = tempA ^ tempB;
    // bankgroup
    tempA = physicalAddress;
    physicalAddress = physicalAddress >> bankgroupBitWidth;
    tempB = physicalAddress << bankgroupBitWidth;
    newTransactionBank |= (tempA ^ tempB) << (bankBitWidth - bankgroupBitWidth);

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> ba_high_width;
    tempB = physicalAddress << ba_high_width;
    newTransactionBank |= (tempA ^ tempB) << ba_low_width;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> col_high_width;
    tempB = physicalAddress << col_high_width;
    newTransactionColumn |= (tempA ^ tempB) << col_low_width;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> rowBitWidth;
    tempB = physicalAddress << rowBitWidth;
    newTransactionRow = tempA ^ tempB;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> rankBitWidth;
    tempB = physicalAddress << rankBitWidth;
    newTransactionRank = tempA ^ tempB;
}

#endif /* _PIM_DUMP_HPP_ */
